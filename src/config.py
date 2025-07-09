"""Config class."""

from pathlib import Path

from src.utils import read_yml

_allowed_exts = [
    ".yml",
    ".yaml",
]


class Config:
    """Config class."""

    def __init__(self, path_root):
        """Initialize config object.

        Args:
            path_root (str): Root config path.

        Raises:
            NameError: Invalid file format.
        """
        self.params = {}
        self.allowed_exts = _allowed_exts

        if Path(path_root).suffix not in self.allowed_exts:
            raise NameError(f"Invalid file format. Expect {self.allowed_exts} for {str(path_root)}")

        self.__add(path_root)
        self.__fill()

    def __add(self, path):
        """Add a config dictionary to inner `params` dictionary.

        Args:
            path (str): Config path.
        """
        config = read_yml(path)
        self.params.update(config)

    def __fill(self):
        """Fill inner `params` dictionary with all config(s) path(s) indexed int root config.

        Raises:
            NameError: Invalid file format.
        """
        for key in self.params["paths_configs"]:
            if Path(self.params["paths_configs"][key]).suffix not in self.allowed_exts:
                raise NameError(
                    f"Invalid file format. Expect {self.allowed_exts} for {self.params['paths_configs'][key]}"
                )

            self.__add(self.params["paths_configs"][key])
