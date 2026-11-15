## Set up prism_explorer on Windows 11

Open a powershell terminal and **ensure you are in the Galaxy-Farm/prism_explorer directory** before running these commands!

One command per line:

`python -m venv .venv`
`.\.venv\Scripts\activate.ps1`
`python -m pip install -r .\requirements.txt`
`streamlit run prism_explorer.py`

## Set up prism_explorer on Linux

One command per line:

`python3 -m venv .venv`
`.venv/bin/python .venv/bin/pip install -r requirements.txt`
`.venv/bin/streamlit run prism_explorer.py`
