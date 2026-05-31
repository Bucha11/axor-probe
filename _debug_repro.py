import asyncio
import sys
import traceback

sys.path.insert(0, ".")
import tests.test_pipeline as t


async def main():
    p, store = t._make_pipeline(comparator=t._HighDriftComparator())
    try:
        r = await p.run(t._EVENT)
        print("report is None:", r is None)
        print("probes_sent:", p._probes_sent)
        print("probes_invalid:", p._probes_invalid)
        print("signals:", len(p._drift_signals))
    except Exception:
        print("EXCEPTION RAISED:")
        traceback.print_exc()


asyncio.run(main())
