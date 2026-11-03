def test_global_chronology_purge_and_coverage(examples):
    _, y, splits, metadata = examples
    train, validation, test = (y.iloc[splits[key]] for key in ["train", "validation", "test"])
    assert train.target_booking_timestamp.max() < validation.target_booking_timestamp.min()
    assert validation.target_booking_timestamp.max() < test.target_booking_timestamp.min()
    assert train.target_booking_timestamp.max() < validation.prediction_cutoff.min()
    assert validation.target_booking_timestamp.max() < test.prediction_cutoff.min()
    assert len(y) == sum(len(x) for x in splits.values()) + metadata["purged_examples"]
    assert all(not set(splits[a]) & set(splits[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
