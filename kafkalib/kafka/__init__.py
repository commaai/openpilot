from __future__ import absolute_import

__title__ = 'kafka'
from kafka.version import __version__
__author__ = 'Dana Powers'
__license__ = 'Apache License 2.0'
__copyright__ = 'Copyright 2016 Dana Powers, David Arthur, and Contributors'

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())


from kafka.admin import KafkaAdminClient
from kafka.client_async import KafkaClient
from kafka.consumer import KafkaConsumer
from kafka.consumer.subscription_state import ConsumerRebalanceListener
from kafka.producer import KafkaProducer
from kafka.conn import BrokerConnection
from kafka.serializer import Serializer, Deserializer
from kafka.structs import TopicPartition, OffsetAndMetadata


__all__ = [
    'BrokerConnection', 'ConsumerRebalanceListener', 'KafkaAdminClient',
    'KafkaClient', 'KafkaConsumer', 'KafkaProducer',
]
