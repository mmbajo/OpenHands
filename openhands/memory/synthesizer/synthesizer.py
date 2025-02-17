import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openhands.core.config import LLMConfig
from openhands.core.message import Message, TextContent
from openhands.events.event import Event
from openhands.events.stream import EventStream


@dataclass
class Environment:
    """Represents the environment for agent interaction"""

    def __init__(self, trajectory: List[Dict[str, Any]]):
        self.trajectory = trajectory
        self.current_idx = 0

    def reset(self):
        """Reset the environment to initial state"""
        self.current_idx = 0

    def get_observation(self) -> str:
        """Get current observation from environment"""
        if self.current_idx >= len(self.trajectory):
            return ''
        return json.dumps(self.trajectory[self.current_idx])

    def finished(self) -> bool:
        """Check if interaction is complete"""
        return self.current_idx >= len(self.trajectory)

    def step(self, action: str) -> str:
        """Take a step in the environment with the given action"""
        if self.finished():
            return ''

        self.current_idx += 1
        return self.get_observation()


class AgentSynthesizer:
    def __init__(
        self,
        llm_config: LLMConfig,
        event_stream: EventStream,
    ):
        self.llm = llm_config.get_llm()
        self.event_stream = event_stream

    def _create_instruction_prompt(self, doc: str) -> Message:
        """Create a prompt for generating instructions from documentation"""
        return Message(
            role='user',
            content=[
                TextContent(
                    text=f'Given this interaction trajectory:\n{doc}\n\n'
                    'Generate a natural language instruction that would lead to this interaction. '
                    'The instruction should be clear, concise, and focused on the task.'
                )
            ],
        )

    def _create_action_prompt(
        self, instruction: str, trajectory: List[tuple[str, str]], observation: str
    ) -> Message:
        """Create a prompt for generating the next action based on current state"""
        history = '\n'.join(
            [f'Observation: {obs}\nAction: {act}' for obs, act in trajectory]
        )

        return Message(
            role='user',
            content=[
                TextContent(
                    text=f'Instruction: {instruction}\n\n'
                    f'Interaction history:\n{history}\n\n'
                    f'Current observation: {observation}\n\n'
                    'What should be the next action to take? '
                    'Respond with just the action description.'
                )
            ],
        )

    def _create_subinstruction_prompt(
        self, trajectory: List[tuple[str, str]]
    ) -> Message:
        """Create a prompt for generating an instruction from a sub-trajectory"""
        history = '\n'.join(
            [f'Observation: {obs}\nAction: {act}' for obs, act in trajectory]
        )

        return Message(
            role='user',
            content=[
                TextContent(
                    text=f'Given this interaction sequence:\n{history}\n\n'
                    'Generate a natural language instruction that would lead to this '
                    'specific sequence of interactions. Be specific and focused on '
                    'this particular sub-task.'
                )
            ],
        )

    def generate_instructions(self, doc: str, num_instructions: int) -> List[str]:
        """Generate multiple instructions from documentation"""
        prompt = self._create_instruction_prompt(doc)
        instructions = []

        for _ in range(num_instructions):
            response = self.llm.chat([prompt])
            instruction = response.choices[0].message.content
            if instruction:
                instructions.append(instruction)

        return instructions

    def generate_action(
        self, instruction: str, trajectory: List[tuple[str, str]], observation: str
    ) -> str:
        """Generate next action based on instruction and current state"""
        prompt = self._create_action_prompt(instruction, trajectory, observation)
        response = self.llm.chat([prompt])
        return response.choices[0].message.content or ''

    def generate_instruction_from_trajectory(
        self, trajectory: List[tuple[str, str]]
    ) -> str:
        """Generate instruction from a trajectory segment"""
        prompt = self._create_subinstruction_prompt(trajectory)
        response = self.llm.chat([prompt])
        return response.choices[0].message.content or ''

    def synthesize(
        self,
        doc: str,
        environment: Environment,
        num_instructions: int,
        data_filter: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Synthesize agent interaction data based on documentation

        Args:
            doc: Standard documentation/instructions or trajectory data
            environment: Environment to interact with
            num_instructions: Number of instructions to generate per document
            data_filter: Optional filter function for generated data

        Returns:
            Dictionary containing synthesized data including instructions and trajectories
        """
        synthesized_data = {'instructions': [], 'trajectories': []}

        # Generate initial instructions from documentation
        instructions = self.generate_instructions(doc, num_instructions)

        for instruction in instructions:
            # Initialize trajectory for this instruction
            trajectory = []
            environment.reset()

            # Generate interaction trajectory
            while not environment.finished():
                observation = environment.get_observation()
                action = self.generate_action(instruction, trajectory, observation)

                # Take step in environment
                next_observation = environment.step(action)

                # Record interaction
                trajectory.append((observation, action))

                # Save to event stream
                self.event_stream.add_event(
                    Event(type='observation', content=observation)
                )
                self.event_stream.add_event(Event(type='action', content=action))

            # Add final observation if any
            if next_observation:
                trajectory.append((next_observation, ''))

            # Generate sub-instructions from trajectory segments
            sub_instructions = []
            for i in range(0, len(trajectory) - 1, 2):
                for j in range(i + 2, len(trajectory), 2):
                    sub_trajectory = trajectory[i:j]
                    sub_instruction = self.generate_instruction_from_trajectory(
                        sub_trajectory
                    )
                    sub_instructions.append(
                        {'instruction': sub_instruction, 'trajectory': sub_trajectory}
                    )

            synthesized_data['instructions'].append(
                {
                    'main_instruction': instruction,
                    'sub_instructions': sub_instructions,
                    'full_trajectory': trajectory,
                }
            )

        # Filter low-quality data if filter provided
        if data_filter:
            synthesized_data = data_filter(synthesized_data)

        return synthesized_data

    def save_synthesized_data(self, data: Dict[str, Any], output_path: str):
        """Save synthesized data to a JSON file

        Args:
            data: Synthesized data dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
