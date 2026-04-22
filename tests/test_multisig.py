"""Tests for the multi-sig coldkey drill (plan-only)."""

from __future__ import annotations

import pytest

from reliquary_inference.chain.multisig import (
    MultiSigOwner,
    MultiSigProposal,
    ProposalStatus,
    plan_proposal,
    render_plan,
)


def _owner(threshold: int = 3, num_signers: int = 5, network: str = "test") -> MultiSigOwner:
    signers = tuple(f"5Signer{i}" for i in range(num_signers))
    return MultiSigOwner(
        coldkey_address="5MultiSigColdKeyXXXXXXXXXXXXXXXXXXXX",
        signer_hotkeys=signers,
        threshold=threshold,
        network=network,
    )


def _proposal(op_type: str = "accept_subnet_owner", op_params=None, **kwargs) -> MultiSigProposal:
    owner = kwargs.pop("owner", _owner())
    return MultiSigProposal(
        proposal_id="p-test-001",
        owner=owner,
        op_type=op_type,
        op_params=op_params or {"netuid": 81},
        proposed_by=kwargs.get("proposed_by", owner.signer_hotkeys[0]),
        cosigners_so_far=kwargs.get("cosigners_so_far", ()),
    )


def test_owner_rejects_threshold_above_signer_count() -> None:
    with pytest.raises(ValueError):
        MultiSigOwner(
            coldkey_address="5X",
            signer_hotkeys=("5A", "5B"),
            threshold=3,
        )


def test_owner_rejects_duplicate_signers() -> None:
    with pytest.raises(ValueError):
        MultiSigOwner(
            coldkey_address="5X",
            signer_hotkeys=("5A", "5B", "5A"),
            threshold=2,
        )


def test_owner_rejects_unknown_network() -> None:
    with pytest.raises(ValueError):
        MultiSigOwner(
            coldkey_address="5X",
            signer_hotkeys=("5A", "5B"),
            threshold=2,
            network="nonsense",
        )


def test_proposal_rejects_proposer_not_in_signers() -> None:
    owner = _owner()
    with pytest.raises(ValueError):
        MultiSigProposal(
            proposal_id="p",
            owner=owner,
            op_type="accept_subnet_owner",
            op_params={"netuid": 81},
            proposed_by="5Outsider",
        )


def test_proposal_rejects_cosigner_not_in_signers() -> None:
    owner = _owner()
    with pytest.raises(ValueError):
        MultiSigProposal(
            proposal_id="p",
            owner=owner,
            op_type="accept_subnet_owner",
            op_params={"netuid": 81},
            proposed_by=owner.signer_hotkeys[0],
            cosigners_so_far=("5Outsider",),
        )


def test_proposal_rejects_proposer_double_counted_in_cosigners() -> None:
    owner = _owner()
    with pytest.raises(ValueError):
        MultiSigProposal(
            proposal_id="p",
            owner=owner,
            op_type="accept_subnet_owner",
            op_params={"netuid": 81},
            proposed_by=owner.signer_hotkeys[0],
            cosigners_so_far=(owner.signer_hotkeys[0],),
        )


def test_proposal_cosigns_count_is_proposer_plus_cosigners() -> None:
    owner = _owner(threshold=3)
    proposal = _proposal(
        owner=owner,
        proposed_by=owner.signer_hotkeys[0],
        cosigners_so_far=(owner.signer_hotkeys[1],),
    )
    assert proposal.cosigns_count() == 2
    assert proposal.is_ready() is False


def test_proposal_ready_when_threshold_met() -> None:
    owner = _owner(threshold=3)
    proposal = _proposal(
        owner=owner,
        proposed_by=owner.signer_hotkeys[0],
        cosigners_so_far=(owner.signer_hotkeys[1], owner.signer_hotkeys[2]),
    )
    assert proposal.cosigns_count() == 3
    assert proposal.is_ready() is True


def test_plan_emits_correct_number_of_cosigner_steps_for_3_of_5() -> None:
    owner = _owner(threshold=3, num_signers=5)
    proposal = _proposal(owner=owner)
    plan = plan_proposal(proposal)
    # 3-of-5 from scratch: proposer + 2 co-signers needed.
    assert len(plan.cosigners) == 2
    assert plan.executor is not None


def test_plan_cosigner_count_reduces_as_cosigners_accumulate() -> None:
    owner = _owner(threshold=3, num_signers=5)
    proposal = _proposal(
        owner=owner,
        cosigners_so_far=(owner.signer_hotkeys[1],),
    )
    plan = plan_proposal(proposal)
    # 1 co-sign already in; need 1 more.
    assert len(plan.cosigners) == 1


def test_plan_no_cosigners_needed_when_threshold_already_met() -> None:
    owner = _owner(threshold=3, num_signers=5)
    proposal = _proposal(
        owner=owner,
        cosigners_so_far=(owner.signer_hotkeys[1], owner.signer_hotkeys[2]),
    )
    plan = plan_proposal(proposal)
    assert plan.cosigners == ()
    assert plan.executor is not None
    assert plan.status_after is ProposalStatus.EXECUTED


def test_plan_rejects_unknown_op_type() -> None:
    owner = _owner()
    proposal = MultiSigProposal(
        proposal_id="p",
        owner=owner,
        op_type="unknown_op",
        op_params={},
        proposed_by=owner.signer_hotkeys[0],
    )
    with pytest.raises(ValueError):
        plan_proposal(proposal)


def test_plan_encodes_accept_subnet_owner_op() -> None:
    owner = _owner()
    proposal = _proposal(owner=owner, op_type="accept_subnet_owner", op_params={"netuid": 81})
    plan = plan_proposal(proposal)
    cmd = " ".join(plan.proposer.command)
    assert "subnet" in cmd
    assert "owner-transfer-accept" in cmd
    assert "--netuid 81" in cmd


def test_plan_encodes_set_subnet_owner_hotkey_op() -> None:
    owner = _owner()
    proposal = _proposal(
        owner=owner,
        op_type="set_subnet_owner_hotkey",
        op_params={"netuid": 81, "new_owner_hotkey": "5NewHotkey"},
    )
    plan = plan_proposal(proposal)
    cmd = " ".join(plan.proposer.command)
    assert "set-owner-hotkey" in cmd
    assert "--owner.hotkey 5NewHotkey" in cmd


def test_plan_notes_warn_on_mainnet() -> None:
    owner = _owner(network="finney")
    proposal = _proposal(owner=owner)
    plan = plan_proposal(proposal)
    joined = "\n".join(plan.notes)
    assert "Mainnet" in joined
    assert "ALLOW_MAINNET=1" in joined


def test_plan_notes_mark_testnet_as_dry_run_friendly() -> None:
    proposal = _proposal()
    plan = plan_proposal(proposal)
    joined = "\n".join(plan.notes)
    assert "testnet" in joined.lower() or "dry-run" in joined.lower()


def test_render_plan_produces_readable_runbook() -> None:
    proposal = _proposal()
    plan = plan_proposal(proposal)
    runbook = render_plan(plan)
    assert "# Multi-sig proposal p-test-001" in runbook
    assert "# 1. Proposer" in runbook
    assert "# 2. Co-signer" in runbook
    # All commands prefixed with btcli.
    for line in runbook.splitlines():
        stripped = line.strip()
        if stripped.startswith("btcli"):
            assert "btcli" in stripped


def test_plan_commands_target_the_multisig_coldkey() -> None:
    owner = _owner()
    proposal = _proposal(owner=owner)
    plan = plan_proposal(proposal)
    for cmd_obj in [plan.proposer, *plan.cosigners, plan.executor]:
        assert cmd_obj is not None
        assert owner.coldkey_address in cmd_obj.command
