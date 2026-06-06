"""Planner, executor, and validator collaboration workflow displayed in DevUI.

The workflow graph intentionally names local nodes, remote nodes, and A2A
handoff nodes so the DevUI block diagram shows how agents coordinate.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, WorkflowViz, handler
from agent_framework.azure import AzureAIClient
from agent_framework.devui import serve
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv


ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)

PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME")

if not PROJECT_ENDPOINT or not MODEL_DEPLOYMENT:
    raise RuntimeError(f"FOUNDRY_PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME are required in {ENV_PATH}")


@dataclass
class UseCasePackage:
    request: str
    use_case_key: str
    use_case_title: str
    a2a_enabled: bool
    routing_notes: str


@dataclass
class PlanPackage:
    request: str
    use_case_title: str
    routing_notes: str
    plan: str


@dataclass
class SolutionPackage:
    request: str
    use_case_title: str
    routing_notes: str
    plan: str
    solution: str
    attempt: int
    feedback: str = ""


@dataclass
class ValidationPackage:
    request: str
    use_case_title: str
    routing_notes: str
    plan: str
    solution: str
    review: str
    approved: bool
    attempt: int


async def create_agent(name: str, instructions: str):
    credential = AzureCliCredential()
    project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)
    conversation = await project_client.get_openai_client().conversations.create()
    chat_client = AzureAIClient(
        project_client=project_client,
        conversation_id=conversation.id,
        model_deployment_name=MODEL_DEPLOYMENT,
    )

    try:
        agent = chat_client.create_agent(name=name, instructions=instructions)
        print(f"Created {name} with conversation {conversation.id}")
        return agent
    finally:
        await chat_client.close()
        await credential.close()


USE_CASES = {
    "1": {
        "title": "A2A remote Azure implementation",
        "notes": "Planner stays local. Executor and validator are modeled as remote A2A agents.",
        "prompt": "Design a production-ready private Azure Storage Account with lifecycle management.",
    },
    "2": {
        "title": "A2A remote code review",
        "notes": "Planner stays local. Remote A2A executor proposes code changes; remote validator reviews them.",
        "prompt": "Review and improve a Python A2A agent executor for reliability and observability.",
    },
    "3": {
        "title": "A2A remote incident response",
        "notes": "Planner stays local. Remote A2A executor drafts mitigation; remote validator checks the runbook.",
        "prompt": "Create an incident response runbook for a failing A2A server hosted on Azure Container Apps.",
    },
    "4": {
        "title": "A2A remote architecture validation",
        "notes": "Planner stays local. Remote A2A executor builds the architecture; remote validator challenges it.",
        "prompt": "Create and validate an architecture for multi-agent collaboration across local and remote agents.",
    },
}


USE_CASE_MENU = "\n".join(
    f"{key}. {value['title']} - {value['prompt']}" for key, value in USE_CASES.items()
)


def parse_use_case(user_input: str) -> tuple[str, str]:
    """Parse a simple DevUI text selection.

    Supported examples:
    - "1"
    - "use_case=2"
    - "2: add private endpoint and monitoring"
    - any free-form request, which defaults to use case 1
    """
    cleaned = user_input.strip()
    if not cleaned:
        return "1", USE_CASES["1"]["prompt"]

    lowered = cleaned.lower()
    for prefix in ("use_case=", "usecase=", "scenario="):
        if lowered.startswith(prefix):
            selected = cleaned[len(prefix) :].strip()
            key, _, custom_request = selected.partition(":")
            key = key.strip()
            return (key if key in USE_CASES else "1"), custom_request.strip() or USE_CASES.get(key, USE_CASES["1"])["prompt"]

    first_token = cleaned.split(":", 1)[0].strip()
    if first_token in USE_CASES:
        _, _, custom_request = cleaned.partition(":")
        return first_token, custom_request.strip() or USE_CASES[first_token]["prompt"]

    return "1", cleaned


class LocalUseCaseSelectorExecutor(Executor):
    @handler
    async def handle(self, user_input: str, ctx: WorkflowContext[UseCasePackage]) -> None:
        use_case_key, request = parse_use_case(user_input)
        use_case = USE_CASES[use_case_key]
        routing_notes = (
            f"Selected use case {use_case_key}: {use_case['title']}\n"
            f"{use_case['notes']}\n\n"
            "A2A protocol path in this diagram:\n"
            "LOCAL_PlannerAgent -> A2A_SendPlanToRemoteExecutor -> "
            "REMOTE_A2A_ExecutorAgent -> A2A_SendSolutionToRemoteValidator -> "
            "REMOTE_A2A_ValidatorAgent."
        )
        await ctx.send_message(
            UseCasePackage(
                request=request,
                use_case_key=use_case_key,
                use_case_title=use_case["title"],
                a2a_enabled=True,
                routing_notes=routing_notes,
            )
        )


class PlannerExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, package: UseCasePackage, ctx: WorkflowContext[PlanPackage]) -> None:
        response = await self.agent.run(
            f"""Create a numbered, dependency-aware plan with risks and acceptance criteria.

Use case:
{package.use_case_title}

Coordination model:
{package.routing_notes}

User request:
{package.request}"""
        )
        await ctx.send_message(
            PlanPackage(
                request=package.request,
                use_case_title=package.use_case_title,
                routing_notes=package.routing_notes,
                plan=str(response),
            )
        )


class A2APlanHandoffExecutor(Executor):
    @handler
    async def handle(self, package: PlanPackage, ctx: WorkflowContext[PlanPackage]) -> None:
        print("A2A handoff: local planner sends the plan to the remote executor agent.")
        await ctx.send_message(package)


class SolutionExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, package: PlanPackage, ctx: WorkflowContext[SolutionPackage]) -> None:
        prompt = f"""Implement this request according to the plan.

Use case:
{package.use_case_title}

A2A routing:
{package.routing_notes}

Request:
{package.request}

Plan:
{package.plan}

Return a complete standalone solution with commands and verification steps."""
        response = await self.agent.run(prompt)
        await ctx.send_message(
            SolutionPackage(
                request=package.request,
                use_case_title=package.use_case_title,
                routing_notes=package.routing_notes,
                plan=package.plan,
                solution=str(response),
                attempt=1,
            )
        )


class A2ASolutionHandoffExecutor(Executor):
    @handler
    async def handle(self, package: SolutionPackage, ctx: WorkflowContext[SolutionPackage]) -> None:
        print("A2A handoff: remote executor sends the solution to the remote validator agent.")
        await ctx.send_message(package)


class A2ARevisionHandoffExecutor(Executor):
    @handler
    async def handle(self, package: SolutionPackage, ctx: WorkflowContext[SolutionPackage]) -> None:
        print("A2A handoff: remote executor sends the revised solution to the final remote validator.")
        await ctx.send_message(package)


class ValidatorExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, package: SolutionPackage, ctx: WorkflowContext[ValidationPackage]) -> None:
        prompt = f"""Validate the proposed solution for correctness, completeness, security,
command consistency, and acceptance criteria. End with exactly DECISION: APPROVED
or DECISION: REVISE, followed by actionable feedback.

Request:
{package.request}

Plan:
{package.plan}

Proposed solution:
{package.solution}"""
        response = str(await self.agent.run(prompt))
        await ctx.send_message(
            ValidationPackage(
                request=package.request,
                use_case_title=package.use_case_title,
                routing_notes=package.routing_notes,
                plan=package.plan,
                solution=package.solution,
                review=response,
                approved="DECISION: APPROVED" in response.upper(),
                attempt=package.attempt,
            )
        )


class RevisionExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, package: ValidationPackage, ctx: WorkflowContext[SolutionPackage]) -> None:
        prompt = f"""Revise the solution using the validator feedback. Return a complete
standalone replacement solution.

Request:
{package.request}

Plan:
{package.plan}

Previous solution:
{package.solution}

Validator feedback:
{package.review}"""
        response = await self.agent.run(prompt)
        await ctx.send_message(
            SolutionPackage(
                request=package.request,
                use_case_title=package.use_case_title,
                routing_notes=package.routing_notes,
                plan=package.plan,
                solution=str(response),
                attempt=2,
                feedback=package.review,
            )
        )


class PublisherExecutor(Executor):
    @handler
    async def handle(self, package: ValidationPackage, ctx: WorkflowContext[str]) -> None:
        status = "APPROVED" if package.approved else "REVIEW REQUIRED AFTER FINAL ATTEMPT"
        output = f"""COLLABORATION RESULT: {status}

SELECTED A2A USE CASE
{package.use_case_title}

LOCAL VS REMOTE AGENT MAP
- LOCAL_UI_UseCaseSelector: local DevUI input and scenario choice
- LOCAL_PlannerAgent: local planning agent
- A2A_SendPlanToRemoteExecutor: A2A protocol handoff marker
- REMOTE_A2A_ExecutorAgent: remote agent that performs implementation work
- A2A_SendSolutionToRemoteValidator: A2A protocol handoff marker
- REMOTE_A2A_ValidatorAgent / REMOTE_A2A_FinalValidator: remote validation agents
- LOCAL_PublishResult: local final response publisher

A2A ROUTING NOTES
{package.routing_notes}

EXECUTION ATTEMPT: {package.attempt}

FINAL SOLUTION
{package.solution}

VALIDATOR REVIEW
{package.review}"""
        await ctx.yield_output(output)


def is_approved(package: ValidationPackage) -> bool:
    return package.approved


def needs_revision(package: ValidationPackage) -> bool:
    return not package.approved


async def build_workflow():
    planner_agent = await create_agent(
        "DevUI-Planner-Agent",
        "You are a local solution planner. Plan work without implementing it. Clearly prepare work for remote A2A agents.",
    )
    executor_agent = await create_agent(
        "DevUI-Remote-A2A-Executor-Agent",
        "You are a remote A2A executor agent. Implement supplied plans with secure, practical commands.",
    )
    validator_agent = await create_agent(
        "DevUI-Remote-A2A-Validator-Agent",
        "You are a remote A2A validator agent. Approve only correct, complete, secure solutions.",
    )

    use_case_selector = LocalUseCaseSelectorExecutor(id="LOCAL_UI_UseCaseSelector")
    planner = PlannerExecutor(planner_agent, id="LOCAL_PlannerAgent")
    a2a_plan_handoff = A2APlanHandoffExecutor(id="A2A_SendPlanToRemoteExecutor")
    executor = SolutionExecutor(executor_agent, id="REMOTE_A2A_ExecutorAgent")
    a2a_solution_handoff = A2ASolutionHandoffExecutor(id="A2A_SendSolutionToRemoteValidator")
    validator = ValidatorExecutor(validator_agent, id="REMOTE_A2A_ValidatorAgent")
    revision = RevisionExecutor(executor_agent, id="REMOTE_A2A_ExecutorRevision")
    a2a_revision_handoff = A2ARevisionHandoffExecutor(id="A2A_SendRevisionToFinalValidator")
    final_validator = ValidatorExecutor(validator_agent, id="REMOTE_A2A_FinalValidator")
    publisher = PublisherExecutor(id="LOCAL_PublishResult")

    workflow = (
        WorkflowBuilder(
            name="A2A Local Remote Planner Executor Validator",
            description=(
                "Choose an A2A use case in the input box, then run a visible local/remote workflow.\n\n"
                "Input examples:\n"
                f"{USE_CASE_MENU}\n\n"
                "Type just 1, 2, 3, or 4, or type 'use_case=2: your custom request'.\n"
                "The block diagram marks LOCAL agents, REMOTE_A2A agents, and A2A handoff nodes."
            ),
        )
        .set_start_executor(use_case_selector)
        .add_edge(use_case_selector, planner)
        .add_edge(planner, a2a_plan_handoff)
        .add_edge(a2a_plan_handoff, executor)
        .add_edge(executor, a2a_solution_handoff)
        .add_edge(a2a_solution_handoff, validator)
        .add_edge(validator, publisher, condition=is_approved)
        .add_edge(validator, revision, condition=needs_revision)
        .add_edge(revision, a2a_revision_handoff)
        .add_edge(a2a_revision_handoff, final_validator)
        .add_edge(final_validator, publisher)
        .build()
    )

    print("Semantic coordination graph:\n", WorkflowViz(workflow).to_mermaid())
    return workflow


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Starting A2A Local/Remote Planner-Executor-Validator DevUI at http://localhost:8092")
    print("Choose an A2A use case in DevUI:")
    print(USE_CASE_MENU)
    workflow = asyncio.run(build_workflow())
    serve(entities=[workflow], port=8092, auto_open=True, tracing_enabled=True)


if __name__ == "__main__":
    main()
