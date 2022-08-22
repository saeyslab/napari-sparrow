#### Testing

### Pytest
In order to run the test suites, we use the package pytest. The config file for pytest is called pytest.ini and is located in the root of the project. 
In the config file the paths of the folders and files that need to be tested are specified.
The commandline arguments read -n=auto for parallel testing and --nbmake for testing the notebooks.

### Executing the tests
You can run the test by executing the command ```pytest``` in the root of the project.