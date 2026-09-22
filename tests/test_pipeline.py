from lotto_nn.pipeline import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    import textwrap

    rows = ["date,n1,n2,n3,n4,n5,n6,bonus"]
    base_numbers = [1, 2, 3, 4, 5, 6]
    for i in range(15):
        nums = [(n + i) % 59 + 1 for n in base_numbers]
        # ensure uniqueness
        nums = sorted(set(nums))
        while len(nums) < 6:
            nums.append(max(nums) + 1 if max(nums) < 59 else 1)
        nums = sorted(nums[:6])
        bonus = 59
        date = f"2022-{(i % 12) + 1:02d}-01"
        rows.append(f"{date},{','.join(str(n) for n in nums)},{bonus}")

    csv_path = tmp_path / "draws.csv"
    csv_path.write_text("\n".join(rows) + "\n")

    results = run_pipeline(
        csv_path=str(csv_path),
        epochs=2,
        batch_size=4,
        num_samples=10,
        verbose=False,
    )

    assert "chi_square" in results
    assert results["generated_np"].shape == (10, 6)
    assert results["sim_hits_max"] >= 0


def test_run_pipeline_writes_output_files(tmp_path):
    rows = ["date,n1,n2,n3,n4,n5,n6,bonus"]
    base_numbers = [1, 2, 3, 4, 5, 6]
    for i in range(15):
        nums = [(n + i) % 59 + 1 for n in base_numbers]
        nums = sorted(set(nums))
        while len(nums) < 6:
            nums.append(max(nums) + 1 if max(nums) < 59 else 1)
        nums = sorted(nums[:6])
        bonus = 59
        date = f"2022-{(i % 12) + 1:02d}-01"
        rows.append(f"{date},{','.join(str(n) for n in nums)},{bonus}")

    csv_path = tmp_path / "draws.csv"
    csv_path.write_text("\n".join(rows) + "\n")

    actual_out = tmp_path / "actual.txt"
    predictions_out = tmp_path / "predictions.txt"
    backtest_out = tmp_path / "backtest.txt"

    results = run_pipeline(
        csv_path=str(csv_path),
        epochs=2,
        batch_size=4,
        num_samples=10,
        verbose=False,
        num_predictions=5,
        actual_out=str(actual_out),
        predictions_out=str(predictions_out),
        backtest_out=str(backtest_out),
    )

    assert actual_out.exists()
    assert predictions_out.exists()
    assert backtest_out.exists()

    actual_line = actual_out.read_text().strip()
    assert len(actual_line.split()) == 6

    prediction_lines = predictions_out.read_text().strip().splitlines()
    assert len(prediction_lines) == 5
    for line in prediction_lines:
        assert len(line.split()) == 6

    assert results["prediction_draws"].shape == (5, 6)
    assert "Chi-square frequency statistic" in backtest_out.read_text()
