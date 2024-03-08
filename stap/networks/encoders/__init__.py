from .base import Decoder, Encoder  # noqa: F401
from .beta_tcvae import VAE  # noqa: F401
from .conv import ConvDecoder, ConvEncoder  # noqa: F401
from .identity import IdentityEncoder  # noqa: F401
from .normalize import NormalizeObservation  # noqa: F401
from .oracle import OracleEncoder  # noqa: F401
from .resnet import ResNet  # noqa: F401
from .table_env import TableEnvEncoder  # noqa: F401

IMAGE_ENCODERS = (ConvEncoder, ResNet, VAE)
