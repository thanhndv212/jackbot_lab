import streamlit as st
import subprocess
import sys
from pathlib import Path
import time
import importlib.util
import pkg_resources


def get_config_path():
    """Get the config file path from the installed package"""
    try:
        # First try to find the package in development mode
        spec = importlib.util.find_spec('jackbot_lab')
        if spec is not None and spec.origin is not None:
            package_root = Path(spec.origin).parent
            config_path = package_root / "tasks" / "locomotion" / "velocity" / "config" / "jackbot" / "flat_env_cfg.py"
            if config_path.exists():
                return config_path
        
        # Fallback to installed package location
        dist = pkg_resources.working_set.by_key['jackbot_lab']
        package_path = Path(dist.location) / "jackbot_lab"
        return package_path / "tasks" / "locomotion" / "velocity" / "config" / "jackbot" / "flat_env_cfg.py"
    except Exception as e:
        st.error(f"Could not find config file: {e}")
        return None


def load_current_weights():
    """Load current weights from flat_env_cfg.py"""
    config_path = get_config_path()
    if config_path is None:
        st.error("Config file not found. Please ensure jackbot_lab package is installed correctly.")
        return {}
        
    weights = {}

    with open(config_path, "r") as f:
        content = f.read()
        # Extract weights using simple parsing
        for line in content.split("\n"):
            if ".weight =" in line:
                try:
                    name = line.split("self.rewards.")[1].split(".weight")[0]
                    weight = float(line.split("=")[1].strip())
                    weights[name] = weight
                except Exception:
                    continue
    return weights


def save_weights(weights):
    """Save weights back to flat_env_cfg.py"""
    config_path = get_config_path()

    with open(config_path, "r") as f:
        content = f.read()

    lines = content.split("\n")
    for i, line in enumerate(lines):
        if ".weight =" in line:
            try:
                name = line.split("self.rewards.")[1].split(".weight")[0]
                if name in weights:
                    lines[i] = (
                        f"        self.rewards.{name}.weight = {weights[name]}"
                    )
            except Exception:
                continue

    with open(config_path, "w") as f:
        f.write("\n".join(lines))


def run_command(command, args, output_placeholder):
    """Run a command with the given arguments in a separate terminal"""
    try:
        # Get the scripts directory relative to this file
        scripts_dir = Path(__file__).parent
        base_dir = scripts_dir.parent.parent

        # Determine which script to run
        if "train.py" in command:
            script_path = scripts_dir / "train.py"
        elif "play.py" in command:
            script_path = scripts_dir / "play.py"
        else:
            script_path = base_dir / command

        # Construct the full command
        full_command = [sys.executable, str(script_path)]
        full_command.extend(args)

        # Create the command string for the terminal
        cmd_str = " ".join(full_command)

        # Create a unique title for the terminal window
        title = (
            f"Jackbot {'Training' if 'train.py' in command else 'Playback'}"
        )

        # Construct the gnome-terminal command
        terminal_cmd = [
            "gnome-terminal",
            "--title",
            title,
            "--",
            "bash",
            "-c",
            f"cd {scripts_dir} && {cmd_str}; echo 'Press Enter to close...'; read",
        ]

        # Run the command in a new terminal
        process = subprocess.Popen(
            terminal_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Update the output placeholder to show that the process started
        output_text = output_placeholder.empty()
        output_text.text(
            f"Started {title} in a new terminal window.\nYou can view the output there."
        )

        return process
    except Exception as e:
        st.error(f"Error running command: {e}")
        return None


def cancel_process(process, output_placeholder):
    """Cancel a running process"""
    if process and process.poll() is None:
        try:
            # Try to terminate the process gracefully
            process.terminate()

            # Wait a bit for graceful termination
            time.sleep(1)

            # If still running, force kill
            if process.poll() is None:
                process.kill()

            # Update the output
            output_text = output_placeholder.empty()
            output_text.text(
                "Process cancelled. Please close the terminal window manually."
            )

            return True
        except Exception as e:
            st.error(f"Error cancelling process: {e}")
            return False
    return False


def main():
    st.title("Reward Weight Tuner")

    # Initialize session state for processes if not already done
    if "training_process" not in st.session_state:
        st.session_state.training_process = None
    if "playback_process" not in st.session_state:
        st.session_state.playback_process = None
    if "training_output" not in st.session_state:
        st.session_state.training_output = None
    if "playback_output" not in st.session_state:
        st.session_state.playback_output = None
    if "show_training_output" not in st.session_state:
        st.session_state.show_training_output = False
    if "show_playback_output" not in st.session_state:
        st.session_state.show_playback_output = False

    # Load current weights
    weights = load_current_weights()

    # Create sections for different types of rewards
    st.header("Velocity Tracking")
    col1, col2 = st.columns(2)
    with col1:
        weights["track_lin_vel_xy_exp"] = st.number_input(
            "Linear Velocity XY",
            value=weights.get("track_lin_vel_xy_exp", 1.2),
            step=0.1,
        )
    with col2:
        weights["track_ang_vel_z_exp"] = st.number_input(
            "Angular Velocity Z",
            value=weights.get("track_ang_vel_z_exp", 0.6),
            step=0.1,
        )

    st.header("Root Penalties")
    col1, col2 = st.columns(2)
    with col1:
        weights["lin_vel_z_l2"] = st.number_input(
            "Linear Velocity Z",
            value=weights.get("lin_vel_z_l2", -0.8),
            step=0.1,
        )
        weights["ang_vel_xy_l2"] = st.number_input(
            "Angular Velocity XY",
            value=weights.get("ang_vel_xy_l2", -0.6),
            step=0.1,
        )
    with col2:
        weights["flat_orientation_l2"] = st.number_input(
            "Flat Orientation",
            value=weights.get("flat_orientation_l2", -0.6),
            step=0.1,
        )
        weights["base_height_exp"] = st.number_input(
            "Base Height", value=weights.get("base_height_exp", -1.5), step=0.1
        )

    st.header("Joint Penalties")
    col1, col2 = st.columns(2)
    with col1:
        weights["joint_deviation_other_l1"] = st.number_input(
            "Joint Deviation Other",
            value=weights.get("joint_deviation_other_l1", -0.3),
            step=0.1,
        )
        weights["joint_deviation_knee_l1"] = st.number_input(
            "Joint Deviation Knee",
            value=weights.get("joint_deviation_knee_l1", -0.2),
            step=0.1,
        )
    with col2:
        weights["joint_deviation_hip_l1"] = st.number_input(
            "Joint Deviation Hip",
            value=weights.get("joint_deviation_hip_l1", -0.2),
            step=0.1,
        )
        weights["joint_deviation_ankle_l1"] = st.number_input(
            "Joint Deviation Ankle",
            value=weights.get("joint_deviation_ankle_l1", -0.2),
            step=0.1,
        )

    st.header("Feet Rewards")
    col1, col2 = st.columns(2)
    with col1:
        weights["feet_air_time"] = st.number_input(
            "Feet Air Time", value=weights.get("feet_air_time", 1.0), step=0.1
        )
        weights["feet_contact"] = st.number_input(
            "Feet Contact", value=weights.get("feet_contact", -0.3), step=0.1
        )
    with col2:
        weights["feet_slide_exp"] = st.number_input(
            "Feet Slide", value=weights.get("feet_slide_exp", -0.5), step=0.1
        )

    st.header("Clock-based Rewards")
    col1, col2 = st.columns(2)
    with col1:
        weights["clock_frc"] = st.number_input(
            "Clock Force", value=weights.get("clock_frc", 0.8), step=0.1
        )
        weights["clock_vel"] = st.number_input(
            "Clock Velocity", value=weights.get("clock_vel", 0.6), step=0.1
        )
    with col2:
        weights["leg_coordination"] = st.number_input(
            "Leg Coordination",
            value=weights.get("leg_coordination", 0.6),
            step=0.1,
        )

    st.header("Gait Rewards")
    col1, col2 = st.columns(2)
    with col1:
        weights["gait_symmetry"] = st.number_input(
            "Gait Symmetry", value=weights.get("gait_symmetry", 0.0), step=0.1
        )
        weights["step_length"] = st.number_input(
            "Step Length", value=weights.get("step_length", 0.4), step=0.1
        )
    with col2:
        weights["air_time_balance"] = st.number_input(
            "Air Time Balance",
            value=weights.get("air_time_balance", 0.6),
            step=0.1,
        )
        weights["feet_alignment"] = st.number_input(
            "Feet Alignment",
            value=weights.get("feet_alignment", 0.5),
            step=0.1,
        )

    # Save button
    if st.button("Save Weights"):
        save_weights(weights)
        st.success("Weights saved successfully!")

    # Display current weights as JSON
    st.header("Current Weights (JSON)")
    st.json(weights)

    # Add a separator
    st.markdown("---")

    # Add training and playback options
    st.header("Training and Playback")

    # Task selection
    st.subheader("Task Selection")
    task_options = {
        "Flat Terrain": "Isaac-Velocity-Flat-Jackbot-v0",
        "Rough Terrain": "Isaac-Velocity-Rough-Jackbot-v0",
    }
    selected_task = st.selectbox(
        "Select Task", options=list(task_options.keys()), index=0
    )
    task_name = task_options[selected_task]

    # Training options
    st.subheader("Training Options")
    train_col1, train_col2 = st.columns(2)

    with train_col1:
        num_envs = st.number_input(
            "Number of Environments", value=4096, min_value=1, step=128
        )
        max_iterations = st.number_input(
            "Max Iterations", value=1000, min_value=100, step=100
        )
        experiment_name = st.text_input(
            "Experiment Name", value="jackbot_velocity"
        )
        run_name = st.text_input("Run Name", value="run_1")

    with train_col2:
        headless = st.checkbox("Headless Mode", value=True)
        resume = st.checkbox("Resume Training", value=False)
        resume_path = st.text_input("Resume Path", value="")
        logger = st.selectbox(
            "Logger", options=["tensorboard", "wandb", "neptune"], index=0
        )

    # Playback options
    st.subheader("Playback Options")
    play_col1, play_col2 = st.columns(2)

    with play_col1:
        checkpoint_path = st.text_input("Checkpoint Path", value="")
        play_experiment_name = st.text_input(
            "Playback Experiment Name", value="jackbot_velocity"
        )
        play_run_name = st.text_input("Playback Run Name", value="run_1")

    with play_col2:
        play_headless = st.checkbox("Playback Headless", value=False)
        record = st.checkbox("Record Playback", value=False)
        play_logger = st.selectbox(
            "Playback Logger",
            options=["tensorboard", "wandb", "neptune"],
            index=0,
        )

    # Run buttons
    col1, col2 = st.columns(2)

    with col1:
        # Create a container for training buttons
        train_buttons = st.container()

        # Check if training is running
        is_training_running = (
            st.session_state.training_process is not None
            and st.session_state.training_process.poll() is None
        )

        # Show appropriate button based on state
        if is_training_running:
            if train_buttons.button("Cancel Training", type="primary"):
                if cancel_process(
                    st.session_state.training_process,
                    st.session_state.training_output,
                ):
                    st.session_state.training_process = None
                    st.session_state.show_training_output = False
                    st.success("Training cancelled.")
        else:
            if train_buttons.button("Run Training"):
                # Create a placeholder for the output
                output_placeholder = st.empty()
                output_placeholder.markdown("### Training Output")
                output_text = output_placeholder.empty()
                output_text.text("Starting training...")

                # Store the output placeholder in session state
                st.session_state.training_output = output_placeholder
                st.session_state.show_training_output = True

                # Construct training arguments
                train_args = [
                    "--task",
                    task_name,
                    "--num_envs",
                    str(num_envs),
                    "--max_iterations",
                    str(max_iterations),
                    "--experiment_name",
                    experiment_name,
                    "--run_name",
                    run_name,
                    "--logger",
                    logger,
                ]

                if headless:
                    train_args.append("--headless")

                if resume and resume_path:
                    train_args.extend(["--resume", resume_path])

                # Run the training
                process = run_command(
                    "train.py",
                    train_args,
                    output_placeholder,
                )

                if process:
                    # Store the process in session state
                    st.session_state.training_process = process
                    st.info("Training started. Output will appear below.")

        # Show training output if needed
        if (
            st.session_state.show_training_output
            and st.session_state.training_output
        ):
            st.session_state.training_output.empty()
            st.session_state.training_output.markdown("### Training Output")

    with col2:
        # Create a container for playback buttons
        play_buttons = st.container()

        # Check if playback is running
        is_playback_running = (
            st.session_state.playback_process is not None
            and st.session_state.playback_process.poll() is None
        )

        # Show appropriate button based on state
        if is_playback_running:
            if play_buttons.button("Cancel Playback", type="primary"):
                if cancel_process(
                    st.session_state.playback_process,
                    st.session_state.playback_output,
                ):
                    st.session_state.playback_process = None
                    st.session_state.show_playback_output = False
                    st.success("Playback cancelled.")
        else:
            if play_buttons.button("Run Playback"):
                if not checkpoint_path:
                    st.error("Please provide a checkpoint path")
                else:
                    # Create a placeholder for the output
                    output_placeholder = st.empty()
                    output_placeholder.markdown("### Playback Output")
                    output_text = output_placeholder.empty()
                    output_text.text("Starting playback...")

                    # Store the output placeholder in session state
                    st.session_state.playback_output = output_placeholder
                    st.session_state.show_playback_output = True

                    # Construct playback arguments
                    play_args = [
                        "--task",
                        task_name,
                        "--checkpoint",
                        checkpoint_path,
                        "--experiment_name",
                        play_experiment_name,
                        "--run_name",
                        play_run_name,
                        "--logger",
                        play_logger,
                    ]

                    if play_headless:
                        play_args.append("--headless")

                    if record:
                        play_args.append("--video")

                    # Run the playback
                    process = run_command(
                        "play.py",
                        play_args,
                        output_placeholder,
                    )

                    if process:
                        # Store the process in session state
                        st.session_state.playback_process = process
                        st.info("Playback started. Output will appear below.")

        # Show playback output if needed
        if (
            st.session_state.show_playback_output
            and st.session_state.playback_output
        ):
            st.session_state.playback_output.empty()
            st.session_state.playback_output.markdown("### Playback Output")


if __name__ == "__main__":
    main()
