Monitor an ffmpeg stream recording file from across the network share.

The most common and reliable way to do this is to **periodically check the file's size**. When the size stops changing for a specific period, you can assume the writing process is complete.

-----

### \#\# Monitor File Size for Stability

The logic is simple: check the file size, wait a few seconds, and check it again. If the size is the same for a few consecutive checks, the file is ready to be copied. This method is robust because a growing file is a clear sign that it's still being written to.

Here is a Python script that implements this logic.

```python
import os
import time
import shutil

def wait_for_stable_file_size(file_path, stability_period_sec=15, check_interval_sec=5):
    """
    Waits until the file size of file_path has not changed for a certain period.

    Args:
        file_path (str): The full path to the file to monitor.
        stability_period_sec (int): How long the file size must remain stable (in seconds).
        check_interval_sec (int): How often to check the file size (in seconds).

    Returns:
        bool: True if the file became stable, False if file not found.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Waiting for it to be created...")
        while not os.path.exists(file_path):
            time.sleep(check_interval_sec)
    
    last_size = -1
    stable_time = 0
    
    print(f"Monitoring file: {file_path}")
    
    while stable_time < stability_period_sec:
        try:
            current_size = os.path.getsize(file_path)
        except OSError as e:
            print(f"Error accessing file: {e}. Retrying in {check_interval_sec} seconds...")
            time.sleep(check_interval_sec)
            continue
            
        if current_size == last_size:
            stable_time += check_interval_sec
        else:
            stable_time = 0 # Reset the timer if the size changed
            print(f"File size is changing... (Current size: {current_size} bytes)")

        last_size = current_size
        time.sleep(check_interval_sec)
        
    print(f"File size has been stable for {stability_period_sec} seconds. Assuming completion.")
    return True


# --- Configuration ---
# Use a UNC path for the shared file
source_mp4 = r"\\RECORDING-SERVER\SharedFolder\recording.mp4"
backup_location = r"C:\path\to\your\local_backup_folder"


# Main execution
if wait_for_stable_file_size(source_mp4):
    try:
        print(f"Copying {source_mp4} to {backup_location}...")
        shutil.copy(source_mp4, backup_location)
        print("Copy complete!")
    except Exception as e:
        print(f"An error occurred during copy: {e}")

```

### \#\# How It Works

1.  **`wait_for_stable_file_size` Function**: This is the core of the script.
2.  **Initial Check**: It first waits for the file to exist, in case the script starts before the recording does.
3.  **Monitoring Loop**: The `while` loop continues until the file size has been static for the duration of `stability_period_sec` (e.g., 15 seconds).
4.  **Size Comparison**: Inside the loop, it gets the current file size.
      * If `current_size` is the same as `last_size`, it adds time to the `stable_time` counter.
      * If `current_size` is different, it means `ffmpeg` is still writing data, so it **resets the `stable_time` counter to zero**.
5.  **Completion**: Once the loop exits, it means the file is no longer growing and it's safe to proceed with the copy operation.

You can **adjust the `stability_period_sec` and `check_interval_sec` values** to be more or less conservative based on your network speed and how `ffmpeg` writes its data. For large video files, a stability period of 10-15 seconds is usually very safe.