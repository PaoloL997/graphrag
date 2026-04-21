"""
Test per la persistenza della memoria conversazionale in GraphRAG.

Verifica che le interazioni vengano correttamente salvate in Milvus
anche quando sono meno di LEN_SHORT_MEMORY (soglia normale di overflow).

Struttura dei test
------------------
1. SHORT-TERM ONLY  – dopo N < 5 interazioni la memoria è solo in Redis
2. FLUSH ESPLICITO  – flush_to_long_memory() persiste tutto in Milvus
3. SHUTDOWN         – shutdown() esegue flush + pulizia Redis
4. REINIT AGENT     – simula il reinit della view Django: il vecchio agent
                      viene flushato prima di essere sostituito

Configurazione usata: comm_25078 / Materiali_e_Specifiche
"""

import sys
import os
from typing import cast

# --------------------------------------------------------------------------- #
# Path setup                                                                   #
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCSLM_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "docsLM", "docslm")
for p in (REPO_ROOT, DOCSLM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import redis as redis_lib  # noqa: E402

from graphrag.store.store import Store  # noqa: E402
from graphrag.memory.user_memory import UserMemory, LEN_SHORT_MEMORY  # noqa: E402
from graphrag.memory.manager import MemoryManager  # noqa: E402

# --------------------------------------------------------------------------- #
# Costanti                                                                     #
# --------------------------------------------------------------------------- #
MILVUS_URI = "http://localhost:19530"
DB_NAME = "comm_25078"
COLLECTION = "Materiali_e_Specifiche"
TEST_USER = "test_memory_flush_user"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

INTERACTIONS = [
    (
        "Quali materiali sono richiesti per la commessa?",
        "I materiali richiesti includono acciaio inox AISI 316L e guarnizioni PTFE.",
    ),
    (
        "Qual è la specifica per le flange?",
        "Le flange devono essere conformi alla norma ASME B16.5 classe 150.",
    ),
    (
        "Sono previsti trattamenti superficiali?",
        "Sì, è richiesta la sabbiatura Sa 2.5 e verniciatura epossidica a 2 mani.",
    ),
    (
        "Che tolleranze dimensionali si applicano?",
        "Tolleranze ISO 2768-m per le lavorazioni meccaniche generali.",
    ),
]

assert len(INTERACTIONS) < LEN_SHORT_MEMORY, (
    "Le interazioni di test devono essere < LEN_SHORT_MEMORY per testare il flush manuale"
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _redis_len(user: str) -> int:
    r = redis_lib.StrictRedis(host="localhost", port=6379, decode_responses=True)
    return cast(int, r.llen(user))


def _milvus_len(user_memory: UserMemory) -> int:
    """Conta i documenti salvati in Milvus per questo utente."""
    try:
        store = user_memory.long_memory_store
        docs = store.vector_store.similarity_search("memoria test", k=100)
        return len(docs)
    except Exception:
        return 0


def _cleanup(user_memory: UserMemory) -> None:
    """Rimuove la collection Milvus e la chiave Redis di test."""
    print(f"\n{YELLOW}[CLEANUP] Rimozione dati di test...{RESET}")
    try:
        from graphrag.store.store import drop_collection

        drop_collection(
            uri=MILVUS_URI,
            database="memory",
            collection=TEST_USER,
        )
        r = redis_lib.StrictRedis(host="localhost", port=6379, decode_responses=True)
        r.delete(TEST_USER)
        print(f"{GREEN}[CLEANUP] Completato.{RESET}")
    except Exception as e:
        print(f"{RED}[CLEANUP] Errore: {e}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")
    raise AssertionError(msg)


def check(condition: bool, success: str, error: str) -> None:
    ok(success) if condition else fail(error)


# --------------------------------------------------------------------------- #
# Test 1 – Short-term only                                                     #
# --------------------------------------------------------------------------- #


def test_short_term_only(user_memory: UserMemory) -> None:
    print(f"\n{'=' * 60}")
    print(f"TEST 1 – Dopo {len(INTERACTIONS)} interazioni la memoria è solo in Redis")
    print(f"{'=' * 60}")

    for q, r in INTERACTIONS:
        user_memory.add(query=q, response=r)

    redis_count = _redis_len(TEST_USER)
    milvus_count = _milvus_len(user_memory)

    print(f"  Redis entries : {redis_count}")
    print(f"  Milvus entries: {milvus_count}")

    check(
        redis_count == len(INTERACTIONS),
        f"Redis contiene {redis_count} entries (atteso {len(INTERACTIONS)})",
        f"Redis dovrebbe avere {len(INTERACTIONS)} entries, trovate {redis_count}",
    )
    check(
        milvus_count == 0,
        "Milvus è ancora vuoto (nessun overflow automatico)",
        f"Milvus non dovrebbe avere entries, trovate {milvus_count}",
    )


# --------------------------------------------------------------------------- #
# Test 2 – Flush esplicito                                                     #
# --------------------------------------------------------------------------- #


def test_explicit_flush(user_memory: UserMemory) -> None:
    print(f"\n{'=' * 60}")
    print("TEST 2 – flush_to_long_memory() persiste tutto in Milvus")
    print(f"{'=' * 60}")

    user_memory.flush_to_long_memory()

    redis_count = _redis_len(TEST_USER)
    milvus_count = _milvus_len(user_memory)

    print(f"  Redis entries (dopo flush): {redis_count}")
    print(f"  Milvus entries            : {milvus_count}")

    check(
        redis_count == len(INTERACTIONS),
        "Redis mantiene le entries dopo il flush (flush non cancella Redis)",
        f"Atteso {len(INTERACTIONS)} entries in Redis, trovate {redis_count}",
    )
    check(
        milvus_count > 0,
        f"Milvus ora contiene {milvus_count} entries",
        "Milvus dovrebbe contenere le entries dopo il flush",
    )


# --------------------------------------------------------------------------- #
# Test 3 – Shutdown                                                            #
# --------------------------------------------------------------------------- #


def test_shutdown() -> None:
    print(f"\n{'=' * 60}")
    print("TEST 3 – shutdown() esegue flush + pulizia Redis")
    print(f"{'=' * 60}")

    manager = MemoryManager(uri=MILVUS_URI)

    # Aggiungi alcune interazioni tramite il manager
    for q, r in INTERACTIONS[:2]:
        manager.save(TEST_USER + "_shutdown", q, r)

    redis_before = _redis_len(TEST_USER + "_shutdown")
    print(f"  Redis entries prima di shutdown: {redis_before}")

    manager.shutdown()

    redis_after = _redis_len(TEST_USER + "_shutdown")
    print(f"  Redis entries dopo shutdown    : {redis_after}")

    check(
        redis_before == 2,
        f"Redis aveva {redis_before} entries prima dello shutdown",
        f"Atteso 2 entries in Redis prima dello shutdown, trovate {redis_before}",
    )
    check(
        redis_after == 0,
        "Redis è vuoto dopo shutdown",
        f"Redis dovrebbe essere vuoto dopo shutdown, trovate {redis_after}",
    )

    # Cleanup Milvus per utente _shutdown
    from graphrag.store.store import drop_collection

    try:
        drop_collection(
            uri=MILVUS_URI, database="memory", collection=TEST_USER + "_shutdown"
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Test 4 – Reinit agent (simula initialize_agent Django)                      #
# --------------------------------------------------------------------------- #


def test_reinit_agent() -> None:
    print(f"\n{'=' * 60}")
    print("TEST 4 – Reinit agent: la memoria del vecchio agent viene flushata")
    print(f"{'=' * 60}")

    store = Store(
        uri=MILVUS_URI,
        database=DB_NAME,
        collection=COLLECTION,
        k=4,
    )

    # Simula il ciclo di vita di initialize_agent
    from graphrag.graph.agent import GraphRAG
    from graphrag.config.prompts import PromptsConfig

    # Prompts minimi (non verranno usati per la query, solo per inizializzare)
    prompts = PromptsConfig(
        generate_response="{context}\n{memory}\n{query}",
        evaluate_context="{context}\n{query}",
        refine_query="{memory}\n{query}",
    )

    user_id_reinit = TEST_USER + "_reinit"

    # --- Primo agent ---
    agent_old = GraphRAG(store=store, llm="gpt-4.1-nano", prompts=prompts)
    for q, r in INTERACTIONS[:3]:
        agent_old.memory_manager.save(user_id_reinit, q, r)

    redis_before = _redis_len(user_id_reinit)
    print(f"  Redis entries prima del reinit: {redis_before}")

    # Simula il flush che avviene in initialize_agent prima di sostituire
    agent_old.memory_manager.flush_all_to_long_memory()

    # Verifica
    user_memory_reinit = agent_old.memory_manager.get_or_create(user_id_reinit)
    milvus_count = _milvus_len(user_memory_reinit)
    print(f"  Milvus entries dopo il flush  : {milvus_count}")

    check(
        redis_before == 3,
        "Redis aveva 3 entries prima del reinit",
        f"Atteso 3 entries in Redis prima del reinit, trovate {redis_before}",
    )
    check(
        milvus_count > 0,
        f"Milvus ha ricevuto {milvus_count} entries dal vecchio agent",
        "Milvus dovrebbe contenere le entries del vecchio agent dopo il flush",
    )

    # Cleanup
    r = redis_lib.StrictRedis(host="localhost", port=6379, decode_responses=True)
    r.delete(user_id_reinit)
    from graphrag.store.store import drop_collection

    try:
        drop_collection(uri=MILVUS_URI, database="memory", collection=user_id_reinit)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main() -> None:
    print(f"\n{YELLOW}{'=' * 60}")
    print("  GRAPHRAG – Test memoria conversazionale")
    print(f"  DB: {DB_NAME}  |  Collection: {COLLECTION}")
    print(f"{'=' * 60}{RESET}")

    user_memory = UserMemory(uri=MILVUS_URI, user=TEST_USER)
    passed = 0
    failed = 0

    tests = [
        ("SHORT-TERM ONLY", lambda: test_short_term_only(user_memory)),
        ("FLUSH ESPLICITO", lambda: test_explicit_flush(user_memory)),
        ("SHUTDOWN", test_shutdown),
        ("REINIT AGENT", test_reinit_agent),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  {RED}FALLITO: {e}{RESET}")
            failed += 1
        except Exception as e:
            print(f"  {RED}ERRORE INATTESO in {name}: {e}{RESET}")
            failed += 1

    _cleanup(user_memory)

    print(f"\n{YELLOW}{'=' * 60}")
    print(f"  RISULTATO: {passed} passati, {failed} falliti")
    print(f"{'=' * 60}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
