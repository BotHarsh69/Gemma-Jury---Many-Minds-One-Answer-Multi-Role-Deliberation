import os
import sys

def load_credentials():
    # 1. Check existing os.environ
    if os.environ.get("GEMINI_API_KEY"):
        return

    # 2. Check local .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k == "GEMINI_API_KEY" and v:
                            os.environ["GEMINI_API_KEY"] = v
                            return
        except Exception:
            pass

    # 3. Check Windows Registry (HKCU / HKLM) in case it was set via setx or Windows GUI
    if sys.platform == "win32":
        try:
            import winreg
            # Check User environment
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                    val, _ = winreg.QueryValueEx(key, "GEMINI_API_KEY")
                    if val:
                        os.environ["GEMINI_API_KEY"] = val
                        return
            except FileNotFoundError:
                pass

            # Check System environment
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                    val, _ = winreg.QueryValueEx(key, "GEMINI_API_KEY")
                    if val:
                        os.environ["GEMINI_API_KEY"] = val
                        return
            except FileNotFoundError:
                pass
        except Exception:
            pass

def test_api():
    load_credentials()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY could not be found in process environment, Windows registry, or .env file.", file=sys.stderr)
        sys.exit(1)

    from google import genai

    # Initialize google-genai client with GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    
    model_name = "gemma-4-26b-a4b-it"
    prompt = "Explain in one sentence why independent opinions can improve reliability."
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        print("--- API CALL SUCCESSFUL ---")
        print(response.text.strip())
    except Exception as e:
        print(f"API Call Failed: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    test_api()
