



class Store:
    def __init__(
            self,
            uri: str,
            database: str,
            collection: str,
            embedding_model: str,
            drop_old: bool = False,
        ):