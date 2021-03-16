from setuptools import setup, find_packages

setup (
    name             = "typodetect",
    version          = "0.1",
    description      = "Example application to be deployed.",
    packages         = find_packages(),
    install_requires = [ 'pytz>=2019.3', "grpcio>=1.23.0", "fire>=0.2.1", "protobuf>=3.11.3", "uwsgidecorators>=1.1.0", "flask>=1.1.1", "PyYAML>=5.1.2", "Cython>=0.29.17", "pathos>=0.2.6", "pypinyin>=0.38.1", "gensim>=3.8.3", "configparser>=5.0.0", "aiohttp>=3.6.2", "attaapi>=1.0.0", "asyncio>=3.4.3", "trpc_report_api_python>=0.1.3"],
)
