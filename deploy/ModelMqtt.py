from BydMqtt import BydMqtt, MqttConfig
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np
from threading import Thread
from PIL import Image
import io
import orjson


@dataclass
class ModelMqttConfig(MqttConfig):
    pub_interval: float = 0.03
    host: str = "localhost"
    port: int = 1883
    sub_topic: str = "/observations"
    pub_topic: str = "/actions"


class ModelMqtt(BydMqtt):
    _instance = None

    def __new__(cls, config: ModelMqttConfig):
        """
        Singleton method to create a Isaac mqtt instance.
        """
        if not cls._instance:
            cls._instance = super(ModelMqtt, cls).__new__(cls)
            cls._instance._initialize_mqtt(config)
        return cls._instance

    def _initialize_mqtt(self, config: ModelMqttConfig):
        super().__init__(config)
        self._config = config

    def _pub_msg(self):
        """
        Call by _start_pub_thread to publish data.

        Args:
            data (stringfy json): the input data should be stringfied json.
        """
        while True:
            # print("Ready to publish.")
            time_start = datetime.now()
            if not self._pub_queue.empty() and self._client.is_connected():
                self._client.publish(topic=self._config.pub_topic, payload=self._pub_queue.get())
                # print("Publish one data.")
            else:
                pass
            time_end = datetime.now()
            time_use = (time_end - time_start).total_seconds()
            sleep_time = max(0, self._config.pub_interval - time_use)
            # print(f"time_use: {time_use}")
            # print(f"sleep time: {sleep_time}")
            time.sleep(sleep_time)

    def _start_pub_thread(self):
        self._pub_thread = Thread(target=self._pub_msg)
        self._pub_thread.start()

    def _arrayToJPEG(self, img_array: np.ndarray):
        image = Image.fromarray(img_array)
        byte_io = io.BytesIO()
        image.save(byte_io, format="JPEG")
        jpeg_data = byte_io.getvalue()
        return jpeg_data

    def _DecodeImage(self, json_data: dict):
        """
        Decode the image in json to numpy array

        Args:
        json_data (dict): contains the observations info

        Returns:
        json_data with decoded ndarray image
        """
        # Get the cam list from original data, json_data["cam_list"]
        cam_list: list["str"] = json_data["cam_list"]
        for cam in cam_list:
            # Convert the serialized image to Pillow stream
            cam_bytes = bytes(json_data[cam])
            json_data[cam] = Image.open(io.BytesIO(cam_bytes))
            # Convert Pillow stream to numpy array
            json_data[cam] = np.array(json_data[cam])
        return json_data

    def get_observations(self):
        obs = self.get_message()
        # print(f"obs: {obs}")
        if obs is not None:
            obs = self._DecodeImage(obs)
        return obs

    def enqueue_actions(self, action: np.ndarray) -> None:
        action_dict = {"action": action.tolist()}
        action_serialized = orjson.dumps(action_dict)
        self.enqueue_pub_msg(action_serialized)

    def print_obs_info(self, obs):
        for cam in obs["cam_list"]:
            print(f"right img: {obs[cam].shape}")
        print(f"qpos: {obs['qpos']}")

    def get_action(self):
        msg = self.get_message()
        action = None
        if msg is not None:
            action = msg["action"]
        return action
