# 🚀 Cloud Training Setup: Google Colab

Follow these steps to train your Lunar Lander on Google's cloud GPUs for free.

## 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a **New Notebook**.

## 2. Set Runtime to GPU
- In the top menu, go to **Runtime** > **Change runtime type**.
- Select **T4 GPU** (or any available GPU).
- Click **Save**.

## 3. Paste and Run this Setup Cell
Copy and paste the code below into the first cell of your notebook. This will clone your code, install dependencies, and start the training.

```python
# 1. Install dependencies
!pip install gymnasium[box2d] pygame swig pandas tqdm

# 2. Upload your project files
# (Zip your project folder and upload LunarLanderRL.zip to the Colab file sidebar)

# 3. Create necessary directories
import os
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('core', exist_ok=True)

# 4. Start Training
from train import train
history = train(n_episodes=2000)

# 5. Download your results
from google.colab import files
files.download('models/checkpoint.pth')
files.download('results/training_log.csv')
```

## 4. Monitoring Progress
While training is running, Colab will show a progress bar with your **Average Reward**.
- The environment is "solved" when the average reward hits **200+**.
- Once finished, the code will automatically prompt you to download the trained model (`checkpoint.pth`).
