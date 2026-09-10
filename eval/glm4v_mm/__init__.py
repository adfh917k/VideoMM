from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


from .configuration_glm4v import *
from .modeling_glm4v import *
from .processing_glm4v import *
from .vision_process import *

# if TYPE_CHECKING:
#     from .configuration_glm4v import *
#     from .modeling_glm4v import *
#     from .processing_glm4v import *
#     from .vision_process import *
# else:
#     import sys

#     _file = globals()["__file__"]
#     sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)