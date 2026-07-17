from load_config import load_config

CFG = load_config()
CFG = CFG.get("Risk", {})
print("CFG contents:", CFG)


def rr_ratio(entry: float, stop: float, target: float, action: str, cfg=None) -> bool:
    if cfg is None:
        cfg = CFG
    min_rr = cfg["min_rr_ratio"]
    long = action == "buy"
    if long:
        if not (stop < entry < target):
            return False, (
                f"For a BUY, prices must satisfy stop ({stop}) < entry ({entry}) < target ({target}). "
                "Adjust stop below entry or target above entry."
            )
    else:  # sell/short
        if not (target < entry < stop):
            return False, (
                f"For a SELL, prices must satisfy target ({target}) < entry ({entry}) < stop ({stop}). "
                "Adjust target below entry or stop above entry."
            )

    risk = abs(entry - stop)
    if risk == 0:
        return False, "Risk is zero (entry equals stop). Please set a valid stop loss."
    reward = abs(target - entry)
    rr = reward / risk
    if rr < min_rr:
        return False, (
            f"Risk:reward ratio is {rr:.2f}, below minimum {min_rr}. "
            f"Risk={risk:.2f}, Reward={reward:.2f}. "
            "Either widen the target or tighten the stop to improve the ratio."
        )
    return True, f"Risk:reward ratio {rr:.2f} ≥ {min_rr}"


def run_risk_gate(state: dict, cfg=None) -> dict:
    if cfg is None:
        cfg = CFG
    proposal = state.get("proposal", {})
    if not proposal or proposal.get("action") == "no_trade":
        return {"risk_gate": {"passed": False, "reason": "No trade proposed.", "checks": {}}}
    required = ["entry", "stop", "target", "action"]
    missing = [k for k in required if k not in proposal]
    if missing:
        return {"risk_gate": {
            "passed": False,
            "reason": f"Missing required fields in proposal : {','.join(missing)}",
            "cehcks": {}
        }}
    checks = {}
    reasons = []

    # R-R check
    passed, msg = rr_ratio(
        proposal["entry"], proposal["stop"], proposal["target"], proposal["action"], cfg
    )
    checks["rr_ratio"] = passed
    if not passed:
        reasons.append(f"rr_ratio : {msg}")

    # More checks (position size, volatility, etc.) can be added here
    passed = all(checks.values())
    if passed:
        reason = "All checks passed"
    else:
        reason = " | ".join(reasons)
    print(f"\n\nRisk gate result {reason} \n {checks} \n\n")

    return {
        "risk_gate": {
            "passed": passed,
            "checks": checks,
            "reason": reason
        }

    }
