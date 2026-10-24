# Essential Claude Code Skills for MoFlow Project

This document outlines the most useful Claude Code skills for working on the MoFlow (Video-Conditioned Extension) project. Enabling these skills will enhance your development experience and ensure consistency with project standards.

## Core Development Skills

### /update-config
**Purpose**: Modify project settings.json to configure automated behaviors, environment variables, permissions, and hooks.

**When to Use**:
- Setting up automated code formatting on save
- Configuring environment variables for experiments
- Adding permissions for frequently used commands
- Setting up file watchers or triggers

**Example Usage**:
```
/update-config set FILE_WATCHER_ENABLED=true
/update-config add permission to bash:python fm_eth.py*
```

### /verify
**Purpose**: Test that changes actually work as expected by running the application and observing behavior.

**When to Use**:
- After implementing new features to confirm they work
- When fixing bugs to verify the fix resolves the issue
- Before considering a task complete
- When making changes to core functionality

**Example Usage**:
```
/verify Does the agent-centric video encoder produce correct tensor shapes?
/verify Does tri-modal fusion maintain coordinate synchronization?
```

### /run
**Purpose**: Launch and drive the application to see changes working in real application context (not just tests).

**When to Use**:
- Visualizing training progress
- Checking if video feature extraction works end-to-end
- Testing new configurations
- Demonstrating functionality

**Example Usage**:
```
/run Start training with new agent-centric video features enabled
/run Visualize a sample of extracted agent crops
```

### /code-review
**Purpose**: Review the current diff for correctness bugs and reuse/simplification/efficiency cleanups.

**When to Use**:
- Before committing changes
- During pull request reviews
- When collaborating with others
- After implementing complex features

**Example Usage**:
```
/code-review --effort medium
/code-review --comment to post findings as inline PR comments
/code-review --fix to apply findings to working tree
```

### /simplify
**Purpose**: Review changed code for reuse, simplification, efficiency, and altitude cleanups, then apply the fixes (quality-focused, not bug hunting).

**When to Use**:
- After implementing functionality to clean up the code
- When refactoring complex sections
- Before sharing code with team members
- As part of regular code maintenance

**Example Usage**:
```
/simplify Simplify the agent crop extraction logic
/simplify Improve efficiency of tensor operations in fusion module
```

### /fewer-permission-prompts
**Purpose**: Scan transcripts for common read-only Bash and MCP tool calls, then add a prioritized allowlist to project .claude/settings.json to reduce permission prompts.

**When to Use**:
- When you find yourself repeatedly granting the same permissions
- After identifying frequently used safe commands
- To streamline your development workflow
- When setting up a new development session

**Example Usage**:
```
/fewer-permission-prompts
# Then review and confirm the suggested allowlist
```

### /loop
**Purpose**: Run a prompt or slash command on a recurring interval (e.g. /loop 5m /foo). Omit the interval to let the model self-pace.

**When to Use**:
- Monitoring training progress over time
- Periodically checking if video encoding is complete
- Watching for file changes in data directories
- Recurring validation checks during development

**Example Usage**:
```
/loop 5m Check if training loss is decreasing
/loop Monitor frame index building progress
/loop (self-paced) Wait for video encoding to complete
```

### /keybindings-help
**Purpose**: Customize keyboard shortcuts, rebind keys, add chord bindings, or modify ~/.claude/keybindings.json.

**When to Use**:
- Personalizing your Claude Code experience
- Creating shortcuts for frequent project-specific actions
- Improving workflow efficiency
- Setting up accessibility accommodations

**Example Usage**:
```
/keybindings-help rebind ctrl+s to /run
/keybindings-help add chord shortcut for /verify
/keybindings-help change submit key to cmd+enter
```

## Recommended Skill Combinations

### Development Cycle
```
1. /update-config    - Configure environment for experiment
2. /run              - Start the application/training
3. /verify           - Test that changes work as expected
4. /code-review      - Review code for quality before committing
```

### Refactoring Workflow
```
1. /simplify         - Apply simplification and efficiency fixes
2. /code-review      - Review the simplified code
3. /test             - Run tests to ensure nothing broke
```

### Debugging Process
```
1. /verify           - Formulate hypothesis about what's wrong
2. /run              - Test hypothesis with specific test cases
3. /update-config    - Adjust logging or debugging settings if needed
```

### Maintenance Routine
```
1. /fewer-permission-prompts - Reduce interruptions for common ops
2. /update-config      - Ensure development environment is optimal
```

## Skill-Specific Tips for MoFlow

### /update-config for MoFlow
- Set up automatic black/isort formatting on Python file save:
  ```
  /update-config set python.formatter=black
  /update-config set python.importSorter=isort
  ```
- Configure permissions for frequently used commands:
  ```
  /update-config add permission to bash:python fm_eth.py*
  /update-config add permission to bash:python -m video_encoder.*
  ```
- Set environment variables for experiment tracking:
  ```
  /update-config set env.WANDB_PROJECT=moflow-video-enc
  /update-config set env.PYTHONPATH=${PYTHONPATH}:${workspaceFolder}
  ```

### /verify for MoFlow
Good verification questions for this project:
- "Does the agent-centric video encoder output tensors of shape [B, A, T_obs, D]?"
- "Does tri-modal fusion prevent cross-agent visual leakage through proper identity masking?"
- "Are agent crops properly aligned with agent trajectories using world_to_pixel conversion?"
- "Does disabling USE_AGENT_VIDEO and USE_TRI_MODAL_FUSION fall back to original behavior?"
- "Do tensor shapes remain consistent throughout the pipeline when features are enabled/disabled?"

### /run for MoFlow
Useful run commands:
- "Train Flow Matching model with agent-centric video: python fm_eth.py --cfg cfg/eth_ucy/cor_fm.yml --subset eth"
- "Build frame index for video features: python -m video_encoder.scripts.build_frame_index --data-root /path/to/raw --pkl-dir /path/to/processed --scene eth --split train"
- "Run integrated encoder tests: python test_integrated_encoder.py"
- "Visualize sample agent crops from a batch"

## Getting Started with These Skills

1. **Begin a session**: Start with `/fewer-permission-prompts` to reduce interruptions
2. **Configure environment**: Use `/update-config` to set up your preferred settings
3. **Implement changes**: Make your code modifications
4. **Verify constantly**: Use `/verify` frequently during development to ensure correctness
5. **Test end-to-end**: Use `/run` to see changes working in the full application
6. **Review before commit**: Use `/code-review` and `/simplify` to polish your changes
7. **Monitor progress**: Use `/loop` for long-running operations like training

---

*This guide complements the CLAUDE.md file by focusing specifically on the Claude Code skills that will make you most effective when working on this project.*