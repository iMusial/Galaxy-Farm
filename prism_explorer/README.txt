-== To set up prism_explorer.py on Windows 11 run these commands: ==-
[One command per line]

python -m venv .venv
.\.venv\Scripts\activate.ps1
python -m pip install -r .\requirements.txt
streamlit run prism_explorer.py

-== To set up prism_explorer.py on Linux run these commands ==-

python3 -m venv .venv
.venv/bin/python .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run prism_explorer.py
