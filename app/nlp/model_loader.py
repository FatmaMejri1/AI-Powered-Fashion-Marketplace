class NLPModelLoader:
    @classmethod
    def get_model(cls):
        """
        Deep learning is disabled. Returns None to ensure fallback logic.
        """
        return None

def get_shared_model():
    return NLPModelLoader.get_model()
