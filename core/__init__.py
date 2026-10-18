"""RouteWise core modules.

VERSION is bumped whenever app.py starts depending on something new in this
package. app.py checks it on startup, so a half-finished upload (new app.py,
old core/) reports exactly which files are stale instead of crashing with a
redacted AttributeError on Streamlit Cloud.
"""

VERSION = 3          # 3 = warehouse terminology, Start here tab
