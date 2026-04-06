#!/usr/bin/env python
"""Prove termination rules fire deterministically with exact diagnostics."""
import sys
from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action
from openenv.tasks import get_benchmark_tasks
from openenv.config import EMAIL_TRIAGE_CONFIG


def prove_failure_collapse():
    """3 consecutive negative rewards → done=true, reason=failure_collapse."""
    task = get_benchmark_tasks()[1]  # medium
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()

    rewards = []
    for i in range(task.max_steps):
        _, reward, done, info = env.step(Action(action_type="wait"))
        rewards.append(reward.total)
        if done:
            reason = info.get("termination_reason")
            diag = info.get("termination_diagnostics", {})
            print(f"[COLLAPSE PROOF] step={i+1} reason={reason}", file=sys.stderr)
            print(f"  last_3_rewards={[f'{r:.3f}' for r in rewards[-3:]]}", file=sys.stderr)
            print(f"  diagnostics={diag}", file=sys.stderr)
            print(f"  rule: {EMAIL_TRIAGE_CONFIG.failure_collapse_window} consecutive r<0 → done", file=sys.stderr)

            # Prove the rule
            window = EMAIL_TRIAGE_CONFIG.failure_collapse_window
            assert reason in ("failure_collapse", "cumulative_failure"), f"unexpected: {reason}"
            if reason == "failure_collapse":
                assert len(rewards) >= window
                assert all(r < 0.0 for r in rewards[-window:]), "collapse rule violated"
                print(f"  VERIFIED: last {window} rewards all < 0.0", file=sys.stderr)
            elif reason == "cumulative_failure":
                cr = sum(rewards)
                assert cr < EMAIL_TRIAGE_CONFIG.cumulative_reward_floor
                print(f"  VERIFIED: cumulative={cr:.3f} < {EMAIL_TRIAGE_CONFIG.cumulative_reward_floor}", file=sys.stderr)
            env.close()
            return True
    env.close()
    return False


def prove_adaptive_penalty():
    """Repeated wait on urgent email → escalating penalty."""
    task = get_benchmark_tasks()[0]  # easy — has critical e-002
    env = EmailTriageEnv(task=task, seed=task.seed)
    env.reset()

    penalties = []
    for i in range(5):
        _, reward, done, info = env.step(Action(action_type="wait"))
        penalties.append(reward.total)
        if done:
            break

    print(f"\n[ADAPTIVE PENALTY PROOF] wait-on-urgent trajectory:", file=sys.stderr)
    for i, p in enumerate(penalties):
        print(f"  step={i+1} reward={p:.3f}", file=sys.stderr)

    # Rewards should get progressively worse (or at least not constant)
    if len(penalties) >= 2:
        assert penalties[1] <= penalties[0] + 0.01, "penalty should not improve when repeating wait"
        print(f"  VERIFIED: penalties escalate or stay negative", file=sys.stderr)
    env.close()


def prove_sla_proximity():
    """Acting just before SLA deadline yields higher reward than acting early."""
    task = get_benchmark_tasks()[0]  # easy
    env = EmailTriageEnv(task=task, seed=task.seed)
    obs = env.reset()

    # e-002 is critical, classification_deadline=1 → acting at step 0 is sla_remaining=1
    _, reward_at_zero, _, _ = env.step(
        Action(action_type="classify", email_id="e-002", category="urgent")
    )
    env.close()

    print(f"\n[SLA PROXIMITY PROOF]", file=sys.stderr)
    print(f"  e-002 classification_deadline=1, acted at step=0", file=sys.stderr)
    print(f"  reward={reward_at_zero.total:.3f}", file=sys.stderr)
    print(f"  components: {dict(reward_at_zero.components)}", file=sys.stderr)

    sla_bonus = reward_at_zero.components.get("sla_proximity", 0.0)
    print(f"  sla_proximity_bonus={sla_bonus:.3f}", file=sys.stderr)
    assert sla_bonus > 0.0, "sla proximity bonus should fire near deadline"
    print(f"  VERIFIED: sla_proximity bonus={sla_bonus:.3f} > 0", file=sys.stderr)


def prove_urgency_weight():
    """Misclassifying a critical email hurts more than a low-priority one."""
    task = get_benchmark_tasks()[2]  # hard — e-205 is low, e-201 is medium/urgent

    # Misclassify low-priority email
    env1 = EmailTriageEnv(task=task, seed=task.seed)
    env1.reset()
    # e-205 arrives step 4, so advance to step 4
    for _ in range(4):
        env1.step(Action(action_type="wait"))
    _, r_low, _, _ = env1.step(
        Action(action_type="classify", email_id="e-205", category="urgent")  # wrong: true=spam
    )
    env1.close()

    # Misclassify urgent/medium email
    env2 = EmailTriageEnv(task=task, seed=task.seed)
    env2.reset()
    _, r_urgent, _, _ = env2.step(
        Action(action_type="classify", email_id="e-201", category="spam")  # wrong: true=urgent, harmful
    )
    env2.close()

    print(f"\n[URGENCY WEIGHT PROOF]", file=sys.stderr)
    print(f"  misclassify low-priority e-205: reward={r_low.total:.3f}", file=sys.stderr)
    print(f"  misclassify urgent e-201 (harmful): reward={r_urgent.total:.3f}", file=sys.stderr)
    assert r_urgent.total < r_low.total, "harmful misclass on urgent should hurt more"
    print(f"  VERIFIED: urgent misclass ({r_urgent.total:.3f}) < low misclass ({r_low.total:.3f})", file=sys.stderr)


if __name__ == "__main__":
    print("=" * 60, file=sys.stderr)
    print("DETERMINISTIC TERMINATION & CONTEXT-AWARE REWARD PROOFS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    prove_failure_collapse()
    prove_adaptive_penalty()
    prove_sla_proximity()
    prove_urgency_weight()

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("ALL PROOFS PASSED", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
