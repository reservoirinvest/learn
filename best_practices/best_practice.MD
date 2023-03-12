# Best practice recommendations for programming

1. SVG design with an end in mind
2. Break into small modules   
   * One module could be named `support`
3. Make an `.ipynb` (Jupyter notebook) for each module with the module name as a story   
   * Each cell will have its own imports.
   * Arguments will follow after the initial imports. 
4. Build the steps in a linear fashion, with very small functionalities
5. Transfer the functionalities into `function`s and move it to `.py` file   
   * Mark the moved functions with `# Moved to /dir/xxxx.py` comment in Jupyter notebook
6. Use .venv `virtual environment` with a generated `requirements.txt`   
7. Keep a consistent directory structure as follows   
    ```text
    .venv
    .vscode
    config
    data
    docs
    logs
    notebooks
    src
    zz_temp
    .env
    .gitignore
    requirement.txt
    ```

   * the .env would contain full path: `PYTHONPATH="C:/Users/kashi/python/ib/src"`   
   * ...with the workspace settings.json containing `"python.envFile": "${workspaceFolder}/.env"`

8. The `src` directory would have all the programs
9. `.gitignore` would have
    ```text
    # Ignore folders
    __pycache__
    venv
    .venv
    .vscode
    zz*


    # Ignore files
    logs/*.*
    *.cfg
    z*.*
    *.env
    *checkpoint.*
    .ipynb_checkpoints
    ```
