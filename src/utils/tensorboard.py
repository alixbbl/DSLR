from datetime import datetime
import tensorflow as tf

class TensorBoardCallback:
    def __init__(self, base_log_dir):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = base_log_dir / f"run_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created TensorBoard log directory: {self.log_dir}")
        self.writer = tf.summary.create_file_writer(str(self.log_dir))
    
    def log_iteration(self, iteration, cost, house_name):
        with self.writer.as_default():
            tf.summary.scalar(f'{house_name}/cost', cost, step=iteration)
        self.writer.flush()