"""
Custom MASW-YOLO Detection Model.

Defines `MASWDetectionModel` which inherits from
`ultralytics.nn.tasks.DetectionModel` and registers the custom MSCA and
AFPN layers so that a YAML config file can reference them by name.
"""

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import (
    Conv,
    C2f,
    SPPF,
    Detect,
    Concat,
)

from .modules.msca import MSCA
from .modules.afpn import AFPN


def custom_parse_model(cfg: dict, ch: list, verbose: bool = True) -> nn.ModuleList:
    """
    Custom model-parsing helper that maps string layer names (including
    ``'MSCA'`` and ``'AFPN'``) to their corresponding ``nn.Module``
    classes, then calls ``parse_model`` on the ultralytics-style YAML
    configuration.

    This is a drop-in replacement for the normal ``parse_model`` call
    inside ``DetectionModel.__init__`` — it simply adds the extra
    entries to the module dictionary before delegating.

    Args:
        cfg (dict): Model configuration dictionary (from parsed YAML).
        ch (list):  List of input channels for each layer.
        verbose (bool): Whether to print layer information.

    Returns:
        nn.ModuleList: List of ``(ch_out, module, ch_in)`` tuples.
    """
    # Ultralytics' parse_model function uses a mapping called
    # 'modules' (or a dict named 'ops' / 'module_map') to resolve
    # string names into nn.Module classes.
    #
    # We monkey-patch the mapping in the ultralytics namespace so
    # that the standard ``parse_model`` sees our custom layers.
    import ultralytics.nn.modules as ultra_modules
    import ultralytics.nn.tasks as ultra_tasks

    # Register custom modules into the ultralytics namespace
    ultra_modules.MSCA = MSCA
    ultra_modules.AFPN = AFPN

    # Also inject into the module map dict if it exists in
    # DetectionModel's scope
    if hasattr(ultra_tasks, "modules"):
        ultra_tasks.modules["MSCA"] = MSCA
        ultra_tasks.modules["AFPN"] = AFPN

    # Delegate to the standard ultralytics parse_model
    model, save = ultra_tasks.parse_model(cfg, ch, verbose)
    return model, save


class MASWDetectionModel(DetectionModel):
    """
    MASW-YOLO Detection Model.

    Extends ``ultralytics.nn.tasks.DetectionModel`` by registering the
    custom ``MSCA`` and ``AFPN`` layers so they can be used inside a
    YOLO-style model YAML configuration.

    Usage:
        model = MASWDetectionModel("path/to/yolov8n_masw.yaml", nc=10)
    """

    def __init__(
        self,
        cfg: str = "yolov8n.yaml",
        ch: int = 3,
        nc: int = 80,
        verbose: bool = True,
    ):
        """
        Initialise the MASWDetectionModel.

        Args:
            cfg (str): Path to the model YAML configuration file.
            ch (int): Number of input channels (default: 3 for RGB).
            nc (int): Number of detection classes.
            verbose (bool): Whether to print layer information.
        """
        # We do NOT call super().__init__() directly because
        # DetectionModel.__init__ calls parse_model() which won't
        # recognise our custom layers. Instead we replicate the
        # initialisation logic while plugging in our own parsing.

        import yaml  # lazy import

        with open(cfg, encoding="ascii", errors="ignore") as f:
            self.yaml = yaml.safe_load(f)

        # Define model
        ch = [ch]
        self.model, self.save = custom_parse_model(self.yaml, ch, verbose)
        self.names = {i: f"class{i}" for i in range(nc)}
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            def _forward(x):
                return self.forward(x)
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in _forward(torch.zeros(1, ch[0], s, s))]
            )
            self.stride = m.stride
            m.nc = nc
            m.names = self.names
        else:
            self.stride = torch.Tensor([32])

        # Init weights, biases
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise weights of Conv / BatchNorm layers."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif t is nn.BatchNorm2d:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif t is Detect:
                m._initialize_biases()
