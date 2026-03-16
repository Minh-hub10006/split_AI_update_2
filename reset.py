import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_purge(queue="feature_queue")
print("✅ Đã xóa hết message cũ")
connection.close()