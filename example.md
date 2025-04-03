# Customizing the Sequential Thinking Server

The Sequential Thinking MCP server can be customized to suit your specific needs. Below are some ways you can extend or modify the server's behavior.

## Modifying the Thinking Stages

The default thinking stages (Problem Definition, Research, Analysis, Synthesis, Conclusion) can be adjusted by modifying the `ThoughtStage` enum in the server code:

```python
class ThoughtStage(Enum):
    """Customize these stages to match your thinking framework"""
    PROBLEM_DEFINITION = "Problem Definition"
    RESEARCH = "Research"
    ANALYSIS = "Analysis"
    # Add your custom stages
    BRAINSTORMING = "Brainstorming"  # New stage
    EVALUATION = "Evaluation"         # New stage
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
```

## Enhancing the Thought Data Structure

You can extend the `ThoughtData` class to include additional metadata that's relevant to your specific thinking process:

```python
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
    # Add custom fields
    confidence_level: float = 0.0    # How confident you are in this thought
    sources: List[str] = field(default_factory=list)  # References or sources
    impact_score: int = 0            # How impactful this thought is
    
    def validate(self) -> bool:
        """Validate thought data"""
        if self.thought_number < 1:
            raise ValueError("Thought number must be positive")
        if self.total_thoughts < self.thought_number:
            raise ValueError("Total thoughts must be greater or equal to current thought number")
        # Add custom validation
        if self.confidence_level < 0.0 or self.confidence_level > 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")
        if self.impact_score < 0 or self.impact_score > 10:
            raise ValueError("Impact score must be between 0 and 10")
        return True
```

## Adding Persistence

By default, the thought history is stored in memory. You can add persistence to save your thinking sessions:

```python
import json
import os

# Add these functions to your server code
def save_thoughts(filename="thoughts.json"):
    """Save current thought history to a file"""
    global thought_history
    
    serializable_thoughts = []
    for t in thought_history:
        # Convert Enum to string for serialization
        thought_dict = {
            "thought": t.thought,
            "thought_number": t.thought_number,
            "total_thoughts": t.total_thoughts,
            "next_thought_needed": t.next_thought_needed,
            "stage": t.stage.value,  # Convert Enum to string
            "tags": t.tags,
            "axioms_used": t.axioms_used,
            "assumptions_challenged": t.assumptions_challenged
        }
        serializable_thoughts.append(thought_dict)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_thoughts, f, ensure_ascii=False, indent=2)
    
    return {"status": "success", "message": f"Saved {len(thought_history)} thoughts to {filename}"}

def load_thoughts(filename="thoughts.json"):
    """Load thought history from a file"""
    global thought_history
    
    if not os.path.exists(filename):
        return {"status": "error", "message": f"File {filename} not found"}
    
    with open(filename, 'r', encoding='utf-8') as f:
        serialized_thoughts = json.load(f)
    
    # Clear current thoughts
    thought_history = []
    
    # Reconstruct ThoughtData objects
    for t_dict in serialized_thoughts:
        # Convert string back to Enum
        stage = ThoughtStage.from_string(t_dict["stage"])
        
        thought = ThoughtData(
            thought=t_dict["thought"],
            thought_number=t_dict["thought_number"],
            total_thoughts=t_dict["total_thoughts"],
            next_thought_needed=t_dict["next_thought_needed"],
            stage=stage,
            tags=t_dict.get("tags", []),
            axioms_used=t_dict.get("axioms_used", []),
            assumptions_challenged=t_dict.get("assumptions_challenged", [])
        )
        thought_history.append(thought)
    
    return {"status": "success", "message": f"Loaded {len(thought_history)} thoughts from {filename}"}

# Register these as MCP tools
@mcp.tool()
def save_thinking_session(filename: str = "thoughts.json") -> dict:
    """Save the current thinking session to a file"""
    return save_thoughts(filename)

@mcp.tool()
def load_thinking_session(filename: str = "thoughts.json") -> dict:
    """Load a thinking session from a file"""
    return load_thoughts(filename)
```

## Implementing Enhanced Analysis

You can improve the analysis capabilities by adding more sophisticated algorithms for finding related thoughts:

```python
def find_related_thoughts(current_thought: ThoughtData, max_count: int = 3) -> List[ThoughtData]:
    """Find thoughts related to the current one using better heuristics"""
    global thought_history
    
    # Skip the current thought itself
    other_thoughts = [t for t in thought_history if t != current_thought]
    
    # If no other thoughts, return empty list
    if not other_thoughts:
        return []
    
    # Calculate relevance scores
    scored_thoughts = []
    for t in other_thoughts:
        score = 0.0
        
        # Same stage gets a baseline score
        if t.stage == current_thought.stage:
            score += 0.3
            
        # Tag overlap increases score
        common_tags = set(t.tags) & set(current_thought.tags)
        if common_tags:
            score += 0.1 * len(common_tags)
            
        # Content similarity (simple keyword matching)
        # For a more advanced implementation, consider using embeddings or NLP
        for word in current_thought.thought.lower().split():
            if len(word) > 4 and word in t.thought.lower():
                score += 0.05
                
        scored_thoughts.append((t, score))
    
    # Sort by score (highest first) and take the top matches
    scored_thoughts.sort(key=lambda x: x[1], reverse=True)
    return [t for t, score in scored_thoughts[:max_count]]
```

## Creating Custom Prompts

You can add predefined prompts to guide users through specific thinking processes:

```python
@mcp.tool()
def get_thinking_prompt(stage: str) -> dict:
    """Get a prompt to help with a specific thinking stage"""
    try:
        thought_stage = ThoughtStage.from_string(stage)
        
        prompts = {
            ThoughtStage.PROBLEM_DEFINITION: 
                "Define the problem by answering: What exactly are we trying to solve? " 
                "What are the constraints? How will we know when we've succeeded?",
                
            ThoughtStage.RESEARCH: 
                "Consider what information you need to address this problem. " 
                "What data would be helpful? What do experts say about this issue?",
                
            ThoughtStage.ANALYSIS: 
                "Break down the key components of the problem. " 
                "How do different factors interact? What patterns do you notice?",
                
            ThoughtStage.SYNTHESIS: 
                "How do the analyzed components fit together? " 
                "What bigger picture emerges from the analysis?",
                
            ThoughtStage.CONCLUSION: 
                "What conclusions can you draw? What actions should be taken? " 
                "What are the implications of your findings?"
        }
        
        return {
            "stage": thought_stage.value,
            "prompt": prompts.get(thought_stage, "No specific prompt available for this stage.")
        }
        
    except ValueError as e:
        return {"error": str(e)}
```

## Advanced Configuration Options

For more sophisticated customization, you could implement a configuration system:

```python
import yaml

# Default configuration
DEFAULT_CONFIG = {
    "max_thoughts_per_session": 100,
    "enable_auto_save": False,
    "auto_save_interval": 5,  # Save every 5 thoughts
    "auto_save_filename": "autosave_thoughts.json",
    "related_thoughts_count": 3,
    "min_thought_length": 10,
    "max_thought_length": 5000,
}

# Global config
config = DEFAULT_CONFIG.copy()

@mcp.tool()
def load_config(filename: str = "config.yaml") -> dict:
    """Load configuration from a YAML file"""
    global config
    
    try:
        with open(filename, 'r') as f:
            user_config = yaml.safe_load(f)
            
        # Update config with user values
        if user_config and isinstance(user_config, dict):
            config.update(user_config)
            return {"status": "success", "message": f"Loaded configuration from {filename}", "config": config}
        else:
            return {"status": "error", "message": "Invalid configuration format"}
            
    except Exception as e:
        return {"status": "error", "message": f"Failed to load configuration: {str(e)}"}

@mcp.tool()
def get_config() -> dict:
    """Get the current configuration"""
    global config
    return {"config": config}
```

