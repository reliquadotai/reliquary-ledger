"""Multi-sig coldkey drill — plan-only scaffolding for the OTF conviction flow.

Produces the command sequence an operator runs to exercise a
threshold-signature proposal (e.g. accepting subnet ownership on SN81,
updating the subnet-owner hotkey, rotating a validator coldkey) against a
Bittensor testnet multi-sig coldkey. No live chain calls are issued from
this module; everything is a plan the operator executes via ``btcli``.

Intent: the first time a 3-of-5 multi-sig owner-claim transaction goes
live should not be a production event. This module lets the team dry-run
the exact flow, including per-signer subcommands, expected block
confirmation windows, and the status transitions of a proposal.

Spec reference: ``01_TIER1_PRD.md`` Epic 5 chain safety + ``00_STRATEGY.md``
multi-sig coldkey policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProposalStatus(str, Enum):
    PROPOSED = "proposed"
    AWAITING_COSIGNS = "awaiting_cosigns"
    READY_TO_EXECUTE = "ready_to_execute"
    EXECUTED = "executed"


@dataclass(frozen=True)
class MultiSigOwner:
    """Description of a multi-sig coldkey.

    ``coldkey_address`` is the multi-sig account's SS58 on-chain.
    ``signer_hotkeys`` is the ordered list of signer SS58 hotkey addresses.
    ``threshold`` is the minimum number of co-signs required to execute.
    """

    coldkey_address: str
    signer_hotkeys: tuple[str, ...]
    threshold: int
    network: str = "test"

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError("threshold must be >= 1")
        if self.threshold > len(self.signer_hotkeys):
            raise ValueError(
                f"threshold ({self.threshold}) exceeds number of signers ({len(self.signer_hotkeys)})"
            )
        if len(set(self.signer_hotkeys)) != len(self.signer_hotkeys):
            raise ValueError("signer_hotkeys must be unique")
        if self.network not in ("test", "finney", "local"):
            raise ValueError(f"unsupported network {self.network!r}")


@dataclass
class MultiSigProposal:
    """One proposed onchain operation requiring threshold co-signatures."""

    proposal_id: str
    owner: MultiSigOwner
    op_type: str                          # e.g. "accept_subnet_owner" | "set_subnet_owner_hotkey"
    op_params: dict[str, Any]
    proposed_by: str                      # signer hotkey SS58
    cosigners_so_far: tuple[str, ...] = ()
    status: ProposalStatus = ProposalStatus.PROPOSED
    notes: str = ""

    def __post_init__(self) -> None:
        if self.proposed_by not in self.owner.signer_hotkeys:
            raise ValueError(
                f"proposed_by {self.proposed_by!r} is not a registered signer"
            )
        for cosigner in self.cosigners_so_far:
            if cosigner not in self.owner.signer_hotkeys:
                raise ValueError(
                    f"cosigner {cosigner!r} is not a registered signer"
                )
        if self.proposed_by in self.cosigners_so_far:
            raise ValueError(
                "proposer is implicitly counted; do not list them in cosigners_so_far"
            )

    def cosigns_count(self) -> int:
        """Total signatures present: proposer + explicit co-signers."""
        return 1 + len(self.cosigners_so_far)

    def is_ready(self) -> bool:
        return self.cosigns_count() >= self.owner.threshold


@dataclass
class PlannedCommand:
    """One btcli invocation the operator should run."""

    description: str
    signer_hotkey: str
    command: list[str]


@dataclass
class ProposalPlan:
    """The full operator-facing plan for a multi-sig proposal.

    ``commands`` is ordered: first the proposer runs ``proposer``, then
    each co-signer runs their ``cosigner`` step, then any one signer runs
    ``executor`` once threshold is met.
    """

    proposal_id: str
    status_after: ProposalStatus
    proposer: PlannedCommand
    cosigners: tuple[PlannedCommand, ...]
    executor: PlannedCommand | None
    notes: tuple[str, ...] = ()


def _btcli_preamble(owner: MultiSigOwner) -> list[str]:
    return [
        "btcli",
        "--network", owner.network,
        "--chain.network", owner.network,
    ]


def _accept_subnet_owner_args(op_params: dict[str, Any]) -> list[str]:
    netuid = int(op_params["netuid"])
    return [
        "subnet", "owner-transfer-accept",
        "--netuid", str(netuid),
    ]


def _set_subnet_owner_hotkey_args(op_params: dict[str, Any]) -> list[str]:
    netuid = int(op_params["netuid"])
    new_hotkey = str(op_params["new_owner_hotkey"])
    return [
        "subnet", "set-owner-hotkey",
        "--netuid", str(netuid),
        "--owner.hotkey", new_hotkey,
    ]


_OP_TABLE: dict[str, Any] = {
    "accept_subnet_owner": _accept_subnet_owner_args,
    "set_subnet_owner_hotkey": _set_subnet_owner_hotkey_args,
}


def _op_args(proposal: MultiSigProposal) -> list[str]:
    builder = _OP_TABLE.get(proposal.op_type)
    if builder is None:
        raise ValueError(
            f"unknown op_type {proposal.op_type!r}; supported: {sorted(_OP_TABLE)}"
        )
    return builder(proposal.op_params)


def plan_proposal(proposal: MultiSigProposal) -> ProposalPlan:
    """Emit the operator-facing plan to advance this proposal.

    The returned plan lists the exact ``btcli`` invocations each signer
    runs (in order) — propose, co-sign by the remaining threshold-1
    signers, execute — and a status prediction after the full plan
    completes.
    """
    base = _btcli_preamble(proposal.owner)
    op_args = _op_args(proposal)
    multisig_address_flag = ["--multisig.coldkey", proposal.owner.coldkey_address]

    proposer_cmd = PlannedCommand(
        description=f"propose {proposal.op_type} via proposer hotkey",
        signer_hotkey=proposal.proposed_by,
        command=base
        + ["multisig", "propose"]
        + multisig_address_flag
        + ["--proposer.hotkey", proposal.proposed_by, "--proposal.id", proposal.proposal_id]
        + ["--", *op_args],
    )

    remaining_signers = [
        h for h in proposal.owner.signer_hotkeys
        if h != proposal.proposed_by and h not in proposal.cosigners_so_far
    ]
    needed_cosigns = max(0, proposal.owner.threshold - 1 - len(proposal.cosigners_so_far))

    cosigner_cmds: list[PlannedCommand] = []
    for hotkey in remaining_signers[:needed_cosigns]:
        cosigner_cmds.append(
            PlannedCommand(
                description=f"co-sign via signer hotkey {hotkey}",
                signer_hotkey=hotkey,
                command=base
                + ["multisig", "approve"]
                + multisig_address_flag
                + ["--signer.hotkey", hotkey, "--proposal.id", proposal.proposal_id],
            )
        )

    status_after = ProposalStatus.READY_TO_EXECUTE
    executor_cmd: PlannedCommand | None = None
    if needed_cosigns > 0 or proposal.cosigns_count() < proposal.owner.threshold:
        # Threshold will be met after these co-signs; any single signer can then execute.
        executor_cmd = PlannedCommand(
            description="execute once threshold is met",
            signer_hotkey=proposal.proposed_by,
            command=base
            + ["multisig", "execute"]
            + multisig_address_flag
            + ["--signer.hotkey", proposal.proposed_by, "--proposal.id", proposal.proposal_id],
        )
    else:
        status_after = ProposalStatus.EXECUTED
        executor_cmd = PlannedCommand(
            description="execute (threshold already met at plan time)",
            signer_hotkey=proposal.proposed_by,
            command=base
            + ["multisig", "execute"]
            + multisig_address_flag
            + ["--signer.hotkey", proposal.proposed_by, "--proposal.id", proposal.proposal_id],
        )

    notes: list[str] = []
    if proposal.owner.network == "finney":
        notes.append(
            "Mainnet — ALLOW_MAINNET=1 required in the operator shell. "
            "Every step is a signed onchain transaction."
        )
    else:
        notes.append(
            f"Network={proposal.owner.network}. Dry-run friendly; testnet state is expendable."
        )
    notes.append(
        f"threshold={proposal.owner.threshold}-of-{len(proposal.owner.signer_hotkeys)}; "
        f"so far proposer + {len(proposal.cosigners_so_far)} co-signer(s); "
        f"needs {needed_cosigns} more co-sign(s) before execute."
    )

    return ProposalPlan(
        proposal_id=proposal.proposal_id,
        status_after=status_after,
        proposer=proposer_cmd,
        cosigners=tuple(cosigner_cmds),
        executor=executor_cmd,
        notes=tuple(notes),
    )


def render_plan(plan: ProposalPlan) -> str:
    """Render a plan as a human-readable operator runbook snippet."""
    lines: list[str] = [
        f"# Multi-sig proposal {plan.proposal_id}",
        f"status_after_plan: {plan.status_after.value}",
        "",
    ]
    for note in plan.notes:
        lines.append(f"# NOTE: {note}")
    lines.append("")
    lines.append(f"# 1. Proposer ({plan.proposer.signer_hotkey}): {plan.proposer.description}")
    lines.append("   " + " ".join(plan.proposer.command))
    for i, cosigner in enumerate(plan.cosigners, start=2):
        lines.append("")
        lines.append(f"# {i}. Co-signer ({cosigner.signer_hotkey}): {cosigner.description}")
        lines.append("   " + " ".join(cosigner.command))
    if plan.executor is not None:
        step = 2 + len(plan.cosigners)
        lines.append("")
        lines.append(f"# {step}. Executor ({plan.executor.signer_hotkey}): {plan.executor.description}")
        lines.append("   " + " ".join(plan.executor.command))
    lines.append("")
    return "\n".join(lines)
