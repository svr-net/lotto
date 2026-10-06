from lotto_nn.grammar_pipeline import run_grammar_pipeline


def _make_csv(tmp_path):
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
    return csv_path


def test_run_grammar_pipeline_end_to_end(tmp_path):
    csv_path = _make_csv(tmp_path)

    results = run_grammar_pipeline(
        csv_path=str(csv_path),
        num_samples=10,
        verbose=False,
        seed=0,
    )

    assert "chi_square" in results
    assert "grammar_summary" in results
    assert results["generated_np"].shape == (10, 6)
    assert results["holdout"] is None  # all sample draws fall before the default split date


def test_run_grammar_pipeline_writes_output_files(tmp_path):
    csv_path = _make_csv(tmp_path)

    actual_out = tmp_path / "actual.txt"
    predictions_out = tmp_path / "predictions.txt"
    backtest_out = tmp_path / "backtest.txt"

    results = run_grammar_pipeline(
        csv_path=str(csv_path),
        num_samples=10,
        verbose=False,
        num_predictions=5,
        seed=0,
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
    assert "Grammar summary" in backtest_out.read_text()
    assert "Chi-square frequency statistic" in backtest_out.read_text()


def test_run_grammar_pipeline_holdout_scores_later_draws(tmp_path):
    rows = ["date,n1,n2,n3,n4,n5,n6,bonus"]
    for i in range(20):
        nums = sorted({(n + 3 * i) % 59 + 1 for n in (0, 9, 18, 27, 36, 45)})
        rows.append(f"20{22 + i // 10}-{i % 10 + 1:02d}-01,{','.join(map(str, nums))},7")
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text("\n".join(rows) + "\n")

    results = run_grammar_pipeline(csv_path=str(csv_path), num_samples=10, verbose=False, seed=0)

    h = results["holdout"]
    assert h["test_draws"] == 10
    assert sum(h["dist"]) == 10 * 10
