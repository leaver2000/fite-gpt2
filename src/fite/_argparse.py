import argparse
import dataclasses
from typing import Generic, TypedDict, TypeVar, overload

from typing_extensions import Unpack
__all__ = ["ArgumentParser", "argument", "DefaultNamespace", "Argument"]
T = TypeVar("T")
Ns = TypeVar("Ns", bound=argparse.Namespace)


class Argument(TypedDict, total=False):
    action: str
    nargs: str
    type: type
    choices: list[str]
    required: bool
    help: str
    metavar: str
    dest: str
    prefix: str

@overload
def argument(**kwargs: Unpack[Argument]) -> str:...
@overload
def argument(default: T, **kwargs: Unpack[Argument]) -> T:...
def argument(default=None,**kwargs):
    if default is None:
        kwargs["required"] = True
    return dataclasses.field(default=default, metadata=kwargs)
    
class ArgumentParser(Generic[Ns], argparse.ArgumentParser):
    def __init__(self, namespace: Ns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._namespace = namespace
        for field in dataclasses.fields(self._namespace):
            metadata = dict(field.metadata)
            prefix = "" if metadata.pop("required", None) else "--"
            self.add_argument(f"{prefix}{field.name}", **metadata)

    def parse_args(self, *args:str) -> Ns:
        return super().parse_args(*args, namespace=self._namespace)





@dataclasses.dataclass
class DefaultNamespace(argparse.Namespace):
    force: bool = argument(False, action="store_true", help="Force")
    verbose: bool = argument(False, action="store_true", help="Verbose")


