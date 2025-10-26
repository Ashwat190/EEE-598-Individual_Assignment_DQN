import subprocess

# ------------------------
# MAIN SCRIPT FOR ASSIGNMENT
# ------------------------

print("📘 Starting GridWorld Assignment Workflow...\n")

# Step 1: Run the training script
print("Running Training (train.py)...\n")
subprocess.run(["python", "train.py"])

# Step 2: Run the evaluation/test script
print("Training complete. Now running Evaluation (test.py)...\n")
subprocess.run(["python", "test.py"])

print("Done! You have now trained and evaluated your GridWorld agent.")
