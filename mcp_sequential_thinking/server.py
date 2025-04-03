from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

def main():
    """Main function to run the server"""
    mcp.run()

class ThoughtStage(Enum):
    """Basic thinking stages for structured sequential thinking"""
    PROBLEM_DEFINITION = "Problem Definition"
    RESEARCH = "Research"
    ANALYSIS = "Analysis"
    SYNTHESIS = "Synthesis"
    CONCLUSION = "Conclusion"
    
    @classmethod
    def from_string(cls, value: str) -> 'ThoughtStage':
        """Convert a string to a thinking stage"""
        try:
            return cls(value)
        except ValueError:
            valid_stages = ", ".join(stage.value for stage in cls)
            raise ValueError(f"Invalid thinking stage: '{value}'. Valid stages are: {valid_stages}")

@dataclass
class ThoughtData:
    """Data structure for a single thought"""
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    stage: ThoughtStage
    tags: List[str] = field(default_factory=list)
    axioms_used: List[str] = field(default_factory=list)
    assumptions_challenged: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate thought data"""
        if self.thought_number < 1:
            raise ValueError("Thought number must be positive")
        if self.total_thoughts < self.thought_number:
            raise ValueError("Total thoughts must be greater or equal to current thought number")
        return True

mcp = FastMCP("sequential-thinking")

thought_history = []

@mcp.tool()
def process_thought(thought: str, thought_number: int, total_thoughts: int, 
                    next_thought_needed: bool, stage: str, 
                    tags: Optional[List[str]] = None,
                    axioms_used: Optional[List[str]] = None, 
                    assumptions_challenged: Optional[List[str]] = None) -> dict:
    """Add a sequential thought with its metadata.

    Args:
        thought: The content of the thought
        thought_number: The sequence number of this thought
        total_thoughts: The total expected thoughts in the sequence
        next_thought_needed: Whether more thoughts are needed after this one
        stage: The thinking stage (Problem Definition, Research, Analysis, Synthesis, Conclusion)
        tags: Optional keywords or categories for the thought
        axioms_used: Optional list of principles or axioms used in this thought
        assumptions_challenged: Optional list of assumptions challenged by this thought
    """
    global thought_history
    
    try:
        # Convert stage string to enum
        thought_stage = ThoughtStage.from_string(stage)
        
        # Create thought data object with defaults for optional fields
        thought_data = ThoughtData(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            stage=thought_stage,
            tags=tags or [],
            axioms_used=axioms_used or [],
            assumptions_challenged=assumptions_challenged or []
        )
        
        # Validate and store
        thought_data.validate()
        thought_history.append(thought_data)
        
        # Find related thoughts 
        related_thoughts = [t for t in thought_history 
                           if t.stage == thought_data.stage and t != thought_data][:3]
        
        # Create and return the response
        return {
            "thoughtAnalysis": {
                "currentThought": {
                    "thoughtNumber": thought_data.thought_number,
                    "totalThoughts": thought_data.total_thoughts,
                    "nextThoughtNeeded": thought_data.next_thought_needed,
                    "stage": thought_data.stage.value,
                    "tags": thought_data.tags,
                    "timestamp": datetime.now().isoformat()
                },
                "analysis": {
                    "relatedThoughtsCount": len(related_thoughts),
                    "relatedThoughtSummaries": [
                        {
                            "thoughtNumber": t.thought_number,
                            "stage": t.stage.value,
                            "snippet": t.thought[:100] + "..." if len(t.thought) > 100 else t.thought
                        } for t in related_thoughts
                    ]
                },
                "context": {
                    "thoughtHistoryLength": len(thought_history),
                    "currentStage": thought_data.stage.value
                }
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

@mcp.tool()
def generate_summary() -> dict:
    """Generate a summary of the entire thinking process"""
    global thought_history
    
    if not thought_history:
        return {"summary": "No thoughts recorded yet"}
        
    # Group thoughts by stage
    stages = {}
    for thought in thought_history:
        if thought.stage.value not in stages:
            stages[thought.stage.value] = []
        stages[thought.stage.value].append(thought)
        
    # Create summary
    summary = {
        "totalThoughts": len(thought_history),
        "stages": {
            stage: len(thoughts) for stage, thoughts in stages.items()
        },
        "timeline": [
            {
                "number": t.thought_number,
                "stage": t.stage.value
            } for t in sorted(thought_history, key=lambda x: x.thought_number)
        ]
    }
    
    return {"summary": summary}

@mcp.tool()
def clear_history() -> dict:
    """Clear the thought history"""
    global thought_history
    thought_history.clear()
    return {"status": "success", "message": "Thought history cleared"}

if __name__ == "__main__":
    main()