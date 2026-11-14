# Galaxy-Farm
Farmers around the world navigate shifting temperatures, changing rainfall patterns, water shortages, extreme weather, and declining soil health. These pressures make it difficult to choose crop rotations that protect fields, conserve water, and support long‑term resilience. Your challenge is to create a decision‑support tool that uses NASA Earth observations along with local soil information, crop characteristics, and farmer priorities to help farmers explore rotation strategies that could strengthen soil health and adapt their farms to changing conditions.

## SerpApi key setup
The key is **never committed**. `.streamlit/secrets.toml` is listed in `.gitignore`.

1. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` (or edit the one already there).
2. Paste your key: `SERPAPI_KEY = "your-key"`. Save.
3. Check it: `python serp_search.py` should print `SerpApi key found`.
4. In code:
   ```python
   from serp_search import search, have_key
   results = search("Iowa cereal rye termination soybean site:iastate.edu") if have_key() else []
   ```

Each teammate makes their own `secrets.toml`. On Streamlit Community Cloud, paste the key into the app's **Secrets** setting instead.
If a key is ever committed by accident, make a new one in the SerpApi dashboard. Deleting the file does not remove it from git history.
