from dataclasses import dataclass
import paho.mqtt.client as mqtt
import threading
import queue
import json
import uuid


def set_random_id(length: int) -> str:
    """
    Generate random client_id for MqttConfig with uuid

    Args:
        length (int): The length of the return id.

    Returns:
        str: The random uuid for client id
    """
    return str(uuid.uuid4()).replace('-', '')[:length]


@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    pub_topic: str = "/default/pub"
    sub_topic: str = "/default/sub"
    client_id: str = set_random_id(10)
    sub_queue_maxsize: int = 60
    pub_queue_maxsize: int = 60


class BydMqtt():

    def __init__(self, config: MqttConfig) -> None:

        self._config = config
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._sub_queue = queue.Queue(maxsize=self._config.sub_queue_maxsize)
        self._pub_queue = queue.Queue(maxsize=self._config.pub_queue_maxsize)

    def _on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")

        # 根据连接返回码做不同的处理
        if rc == 0:
            print("Connection successful!")
            # 连接成功后，可以开始订阅主题
            client.subscribe(self._config.sub_topic)
        elif rc == 1:
            print("Connection failed: Incorrect protocol version.")
        elif rc == 2:
            print("Connection failed: Invalid client identifier.")
        elif rc == 3:
            print("Connection failed: Server unavailable.")
        elif rc == 4:
            print("Connection failed: Bad user name or password.")
        elif rc == 5:
            print("Connection failed: Not authorized.")
        else:
            print(f"Connection failed with unknown error code {rc}")

    def _on_message(self, client, userdata, msg):
        """
        Callback on message, json loads msg's payload,
        and then enqueue the data to sub_queue.
        When enqueue the data, if the queue is full,
        the oldest data will be dropped.
        """
        data = json.loads(msg.payload)
        if self._sub_queue.full():
            self._sub_queue.get_nowait()
        self._sub_queue.put_nowait(item=data)

    def get_message(self) -> dict | None:
        """
        Get message from sub queue.

        Returns:
            dict | None: If the queue is empty, return None.
        """
        try:
            return self._sub_queue.get_nowait()
        except queue.Empty:
            return None

    def enqueue_pub_msg(self, msg) -> None:
        """
        Publish message with given MqttConfig["pub_topic"] and data.

        Args:
        data (stringfy json): The data should be stringfied json format.
        """
        try:
            self._pub_queue.put_nowait(item=msg)
        except queue.Full:
            print("Pub queue is full, drop the oldest msg")
            self._pub_queue.get_nowait()

    def start(self) -> None:
        """
        Connect to the mqtt broker server. And create a thread for mqtt running in daemon.
        """
        self._client.connect(self._config.host, self._config.port)
        self._daemon_thread = threading.Thread(target=self._client.loop_forever, daemon=True)
        self._daemon_thread.start()

    def stop(self) -> None:
        """
        Disconnect the mqtt client, and stop the thread.
        """
        self._client.disconnect()
        self._daemon_thread.join()
        print("MQTT client stop")
