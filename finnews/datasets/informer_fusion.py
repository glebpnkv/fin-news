"""Heavily inspired by https://huggingface.co/blog/informer"""
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddAgeFeature,
    AddTimeFeatures,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    Transformation,
    VstackFeatures,
    RenameFields,
)
from transformers import InformerConfig


class RollingWindowSampler(InstanceSampler):
    def __call__(self, ts):
        # ts is the entire target array for a single time series
        # self.past_length and self.future_length come from the splitter
        # by default, they are set when the splitter calls .set_context on the sampler

        n = len(ts)
        valid_start = self.min_past
        valid_end = n - self.min_future

        # Return *all* valid start indices for a rolling window
        # (If you want a stride > 1, replace `1` with the stride you want)
        return list(range(valid_start, valid_end + 1, 1))


class InformerFusionDataset:
    def __init__(
        self,
        input_size: int,
        d_model: int,
        context_length: int = 64,
        prediction_length: int = 14,
        dropout: float = 0.1,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        lags_sequence: list[int] | None = None,
        distribution_output: str = "normal",
        scaling: str | bool | None = None,
        freq: str = "1d",
    ):
        if not lags_sequence:
            lags_sequence = [1]
        self.freq = freq

        time_features = time_features_from_frequency_str(freq)

        self.model_config = InformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            input_size=input_size,
            lags_sequence=lags_sequence,
            scaling=scaling,
            num_time_features=len(time_features) + 1,  # Adding one for Age Time Feature
            dropout=dropout,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_model=d_model,
            distribution_output=distribution_output,
        )

    @staticmethod
    def df_to_gluonts_format(
        df: pd.DataFrame,
        articles_agg: torch.Tensor,
        date_col: str = "date",
        symbol_col: str = "symbol",
        value_col: str = "ret",
        freq: str = "d"
    ) -> ListDataset:
        """
        Converts long-format DataFrames of price returns df - with columns [date, symbol, ...] - and a
        into a list of dicts compatible with GluonTS:
          [{"start": <str>, "target": <list>, "item_id": <symbol>}, ...].
        """
        data = []

        # Iterating over stock symbols
        for symbol, group in df.groupby(symbol_col):
            start_timestamp = group[date_col].iloc[0]  # first date in this symbol group
            record = {
                "start": pd.Period(start_timestamp, freq),
                "target": group[value_col].tolist(),
                "articles": articles_agg[group.index.tolist(), :].cpu().numpy().T,
                "item_id": str(symbol),
            }
            data.append(record)

        dataset = ListDataset(data, freq=freq)

        return dataset

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                # step 1: add temporal features based on freq of the dataset
                # these serve as positional encodings
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.model_config.prediction_length,
                ),
                # step 2: add another temporal feature (just a single number)
                # tells the model where in the life the value of the time series is
                # sort of running counter
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.model_config.prediction_length,
                    log_scale=True,
                ),
                # step 3: vertically stack all the temporal features into the key FEAT_TIME
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                ),
                # step 4: rename to match HuggingFace names
                RenameFields(
                    mapping={
                        FieldName.FEAT_TIME: "time_features",
                        FieldName.TARGET: "values",
                        FieldName.OBSERVED_VALUES: "observed_mask",
                    }
                )
            ]
        )

    def create_instance_splitter(
        self,
        mode: str,
        num_instances: float | None = None
    ) -> InstanceSplitter:
        assert mode in ["train", "validation", "test"]

        if not num_instances:
            num_instances = 1.0

        future_length = self.model_config.prediction_length

        instance_sampler = {
            "train": ExpectedNumInstanceSampler(
                num_instances=num_instances,
                min_future=self.model_config.prediction_length
            ),
            "validation": RollingWindowSampler(min_future=future_length),
            "test": RollingWindowSampler(min_future=future_length),
        }[mode]

        return InstanceSplitter(
            target_field="values",
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.model_config.context_length + max(self.model_config.lags_sequence),
            future_length=self.model_config.prediction_length,
            time_series_fields=["time_features", "articles"]
        )

    def create_dataloader(
        self,
        df: pd.DataFrame,
        articles_agg: torch.Tensor,
        batch_size: int = 32,
        mode: str = "train",
        num_instances: float | None = None,
    ):
        train = mode == "train"
        # Preparing data transformation
        transformation = self.create_transformation()
        # Preparing data splitter
        splitter = self.create_instance_splitter(mode=mode, num_instances=num_instances)

        data = self.df_to_gluonts_format(
            df=df,
            articles_agg=articles_agg,
            freq=self.freq
        )
        data = transformation.apply(data, is_train=train)
        data = splitter.apply(data, is_train=train)

        # Batching
        field_names = [
            "past_time_features",
            "past_values",
            "past_articles",
            "future_time_features",
            "future_values",
        ]
        output_type = torch.tensor

        if mode == "test":
            field_names += [ "item_id", "forecast_start"]
            output_type = None

        dataloader = as_stacked_batches(
            data,
            batch_size=batch_size,
            field_names=field_names,
            output_type=output_type,
        )

        return dataloader
