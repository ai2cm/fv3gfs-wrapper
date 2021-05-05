def pytest_addoption(parser):
    parser.addoption("--refdir", action="store", help="directory for reference files")
