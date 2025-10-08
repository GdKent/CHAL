"""
debate_driver.py

This file is the entry point for running a full debate using the DebateController.
It instantiates agents, defines personas, and executes the controller.
"""

# === Imports ===
from chal.agents.openai_agent import OpenAIAgent
from chal.agents import prompts
from chal.orchestrator.debate_controller import DebateController
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.embeddings.embedding_visualizer import BeliefTrajectoryPlotter

# === Debate Topic Configuration ===

topic = "Does free will exist?"

# Define which personas each agent should adopt for this debate
personas = {
    "Agent-Empiricist": prompts.EMPIRICIST,
    "Agent-Rationalist": prompts.RATIONALIST,
    "Agent-Supernaturalist": prompts.SUPERNATURALIST,
    "Agent-Skeptic": prompts.SKEPTIC,
}

# === Create Debate Agents ===

# These are your main LLM-backed debaters.
# The `system_prompt` will be set by the DebateController during Stage 0 (briefing), so it is left blank here.
agent1 = OpenAIAgent(
    model="gpt-4o",
    name="Agent-Empiricist",
    system_prompt=""
)


agent2 = OpenAIAgent(
    model="gpt-4o",
    name="Agent-Supernaturalist",
    system_prompt=""
)

#agent3 = OpenAIAgent(
#    model="gpt-4o",
#    name="Agent-Skeptic",
#    system_prompt=""
#)

agents = [agent1, agent2]

# === Create Debate Controller ===

controller = DebateController(agents=agents, max_rounds=1)

# === Run the Debate ===

results = controller.run(topic=topic, personas=personas)

# Save synthesis as .txt
with open("src/chal/storage/debate_synthesis.txt", "w", encoding="utf-8") as f:
        f.write(results["synthesis"])

# Save full transcript as .txt
with open("src/chal/storage/debate_transcript.txt", "w", encoding="utf-8") as f:
        f.write(results["full_transcript"])

# Save initial beliefs as .txt
with open("src/chal/storage/initial_beliefs.txt", "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(results["initial_positions"]))

# Save final beliefs as .txt
with open("src/chal/storage/final_beliefs.txt", "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(results["final_positions"]))

print("\n================ Council Adjourned ================")



# Load embeddings
tracker = BeliefEmbeddingTracker()
tracker.load_embeddings("src/chal/storage/embeddings.npz")

# Plot the belief embeddings
plotter = BeliefTrajectoryPlotter(n_components=2)
reduced = plotter.reduce_embeddings(tracker.get_all_embeddings())
plotter.plot_trajectories(reduced)

