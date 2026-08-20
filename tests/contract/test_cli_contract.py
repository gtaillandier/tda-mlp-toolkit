from tda_selector.cli import build_parser


def test_cli_accepts_expected_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["input.xyz", "--n-select", "10", "--selection-method", "fps"])
    assert args.structures == "input.xyz"
    assert args.n_select == 10
    assert args.selection_method == "fps"
