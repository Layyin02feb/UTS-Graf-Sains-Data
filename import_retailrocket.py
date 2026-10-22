
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Impor data training RetailRocket ke Neo4j."
    )
    parser.add_argument("-u", "--user", default="neo4j")
    parser.add_argument("-p", "--password", required=True)
    parser.add_argument(
        "-b", "--bolt",
        default="neo4j://127.0.0.1:7687"
    )
    parser.add_argument("-s", "--source", default="data")
    parser.add_argument("-f", "--file", default="training.csv")
    parser.add_argument("-d", "--database", default="neo4j")
    parser.add_argument("--batch-size", type=int, default=10000)
    return parser.parse_args()


def create_constraints(session):
    session.run("""
        CREATE CONSTRAINT visitor_id_unique IF NOT EXISTS
        FOR (v:Visitor)
        REQUIRE v.visitorId IS UNIQUE
    """).consume()

    session.run("""
        CREATE CONSTRAINT item_id_unique IF NOT EXISTS
        FOR (i:Item)
        REQUIRE i.itemId IS UNIQUE
    """).consume()


def import_batch(tx, batch):
    query = """
    UNWIND $batch AS row

    MERGE (v:Visitor {visitorId: row.visitorid})
    MERGE (i:Item {itemId: row.itemid})

    WITH v, i, row

    CALL {
        WITH v, i, row
        WITH v, i, row WHERE row.event = 'view'
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row
        WITH v, i, row WHERE row.event = 'addtocart'
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row
        WITH v, i, row WHERE row.event = 'transaction'
        CREATE (v)-[:BOUGHT {
            timestamp: row.timestamp,
            transactionId: row.transactionid
        }]->(i)
    }
    """

    tx.run(query, batch=batch).consume()


def main():
    args = parse_arguments()

    if args.batch_size <= 0:
        print("Ukuran batch harus lebih besar dari 0.")
        sys.exit(1)

    csv_file_path = os.path.join(args.source, args.file)

    if not os.path.exists(csv_file_path):
        print(f"File tidak ditemukan: {csv_file_path}")
        sys.exit(1)

    driver = None

    try:
        driver = GraphDatabase.driver(
            args.bolt,
            auth=(args.user, args.password)
        )
        driver.verify_connectivity()
        print("Koneksi Neo4j berhasil.")

        with driver.session(database=args.database) as session:
            create_constraints(session)
            print("Constraint berhasil diperiksa.")

            total_rows = sum(
                1 for _ in open(csv_file_path, encoding="utf-8")
            ) - 1

            total_batches = (
                total_rows + args.batch_size - 1
            ) // args.batch_size

            chunks = pd.read_csv(
                csv_file_path,
                chunksize=args.batch_size
            )

            for chunk in tqdm(
                chunks,
                total=total_batches,
                desc="Mengimpor training"
            ):
                chunk["transactionid"] = (
                    chunk["transactionid"]
                    .fillna("")
                    .astype(str)
                )

                chunk = chunk.astype(object).where(
                    pd.notna(chunk), None
                )

                records = chunk.to_dict(orient="records")
                session.execute_write(import_batch, records)

        print("Impor training selesai.")

    except Exception as e:
        print(f"Proses gagal: {e}")
        sys.exit(1)

    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()
