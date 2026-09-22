"""
Dataset Generator: Creates clean, high-quality Computer Science academic PDFs in data/raw/
for testing and running the RAG Document Assistant pipeline.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

def build_pdf(filename, title, pages_content):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Code'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=8
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 10))

    for i, page_sections in enumerate(pages_content):
        if i > 0:
            story.append(PageBreak())
        for heading, body in page_sections:
            if heading:
                story.append(Paragraph(heading, h1_style))
            story.append(Paragraph(body, body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f"Generated: {filename}")


def main():
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Document 1: Operating Systems & Concurrency
    doc1_path = os.path.join(raw_dir, "cs101_operating_systems_concurrency.pdf")
    doc1_pages = [
        # Page 1
        [
            ("1. Introduction to Processes and Threads",
             "In modern operating systems, a process is an instance of a program in execution. It contains the program code, its current activity represented by the program counter, stack, data section, and heap memory. In contrast, a thread is the smallest unit of execution within a process, often termed a lightweight process. Threads belonging to the same process share the process's address space, open file descriptors, and global variables, but each maintains its own private stack and register state."),
            ("2. Process Scheduling & Dispatching",
             "The CPU scheduler selects among the processes in memory that are ready to execute. Common scheduling algorithms include First-Come-First-Served (FCFS), Shortest Job Next (SJN), Priority Scheduling, and Round Robin (RR). In Round Robin scheduling, each ready process is assigned a fixed time slice called a quantum (typically 10 to 100 milliseconds). If a process does not finish within its quantum, it is preempted and returned to the tail of the ready queue. Modern Linux kernels employ the Completely Fair Scheduler (CFS), which uses red-black trees to balance CPU time among tasks proportionally based on nice values.")
        ],
        # Page 2
        [
            ("3. Synchronization Primitives and Race Conditions",
             "When multiple concurrent threads access shared data without synchronization, race conditions occur where the outcome depends on the non-deterministic order of execution. Critical sections are code segments accessing shared resources. Mutual exclusion ensures that only one thread executes in its critical section at any given time. Semaphores, introduced by Edsger Dijkstra, are integer variables accessed via two atomic operations: wait() (or P) and signal() (or V). A binary semaphore behaves like a mutex lock, whereas a counting semaphore allows a fixed number of threads to access a finite pool of resources."),
            ("4. Deadlocks and Coffman Conditions",
             "A deadlock occurs when a set of concurrent processes are blocked because each process is holding a resource and waiting for another resource acquired by some other process. According to Coffman et al. (1971), four conditions must hold simultaneously for a deadlock to arise: 1) Mutual Exclusion: resources cannot be shared; 2) Hold and Wait: processes hold at least one resource while waiting for others; 3) No Preemption: resources cannot be forcibly taken from a process holding them; 4) Circular Wait: a closed chain of processes exists where each process waits for a resource held by the next. Deadlock avoidance algorithms such as Dijkstra's Banker's Algorithm dynamically inspect resource allocation state to ensure safe execution states.")
        ]
    ]
    build_pdf(doc1_path, "CS101: Operating Systems — Concurrency and Process Management", doc1_pages)

    # Document 2: Database Systems & ACID Transactions
    doc2_path = os.path.join(raw_dir, "cs201_database_indexing_and_acid.pdf")
    doc2_pages = [
        # Page 1
        [
            ("1. Relational Storage and Indexing Structures",
             "Database management systems (DBMS) optimize data retrieval using specialized index data structures. The most prevalent indexing mechanism in relational engines (such as PostgreSQL and MySQL InnoDB) is the B+ Tree. Unlike standard binary search trees, B+ trees are self-balancing multi-way search trees with high branching factors (fan-out), ensuring minimal disk I/O operations. In a B+ Tree, all data records or leaf pointers reside exclusively in the leaf nodes, which are linked together sequentially to accelerate range queries. Internal nodes store only search keys and routing pointers."),
            ("2. Clustered vs. Secondary Indexes",
             "A clustered index dictates the physical storage order of table records on disk; therefore, a table can possess only one clustered index (frequently the primary key). Secondary or non-clustered indexes contain secondary search keys paired with pointers (e.g., tuple identifiers or primary key values) that point to the actual table rows. Covering indexes include all columns requested by a specific SQL query, allowing the database query planner to satisfy the query entirely from index leaf nodes without accessing disk pages.")
        ],
        # Page 2
        [
            ("3. The ACID Transaction Model",
             "Transactions are logical units of database work that must adhere to ACID properties: 1) Atomicity: all operations within a transaction succeed or all are rolled back, guaranteed using Write-Ahead Logging (WAL); 2) Consistency: transactions transition the database from one valid state to another according to integrity constraints; 3) Isolation: concurrent transactions execute without interfering with one another; 4) Durability: once committed, transaction results survive system crashes and power failures."),
            ("4. Transaction Isolation Levels and Anomalies",
             "The SQL standard specifies four transaction isolation levels to balance consistency and concurrency performance: Read Uncommitted (allows dirty reads, non-repeatable reads, and phantom reads); Read Committed (prevents dirty reads by acquiring short-term read locks); Repeatable Read (prevents dirty and non-repeatable reads by holding read and write locks until commit time); and Serializable (highest level, eliminates all anomalies including phantom reads, typically enforced via Multi-Version Concurrency Control (MVCC) or Strict Two-Phase Locking (SS2PL)). A phantom read occurs when a transaction queries a range of rows twice and discovers new rows inserted by another committed transaction.")
        ]
    ]
    build_pdf(doc2_path, "CS201: Database Systems — Indexing, Storage, and ACID Transactions", doc2_pages)

    # Document 3: Computer Networking & Protocols
    doc3_path = os.path.join(raw_dir, "cs301_computer_networking_and_protocols.pdf")
    doc3_pages = [
        # Page 1
        [
            ("1. Network Architecture and the Transport Layer",
             "Computer networks operate across layered architectures, notably the 7-layer OSI model and the 4-layer TCP/IP protocol suite. At the transport layer, the Transmission Control Protocol (TCP) offers reliable, connection-oriented byte-stream delivery with sequence numbering, acknowledgment packets, and checksum verification. The User Datagram Protocol (UDP) provides connectionless, lightweight, best-effort datagram delivery without retransmission, ideal for real-time media streaming and online gaming."),
            ("2. The TCP 3-Way Handshake and Connection Teardown",
             "To establish a reliable TCP connection, endpoints execute a 3-Way Handshake: 1) Client sends SYN packet with an initial sequence number (ISN_c); 2) Server responds with SYN-ACK packet containing server ISN_s and ACK = ISN_c + 1; 3) Client replies with ACK packet with ACK = ISN_s + 1. Connection teardown uses a 4-Way Handshake (FIN, ACK, FIN, ACK) with a TIME_WAIT state (usually 2MSL = 2 minutes) on the active closing side to ensure delayed segments in the network do not corrupt subsequent new connections.")
        ],
        # Page 2
        [
            ("3. TCP Congestion Control Algorithms",
             "TCP manages network congestion using mechanisms defined by Van Jacobson: Slow Start, Congestion Avoidance, Fast Retransmit, and Fast Recovery. During Slow Start, the congestion window (cwnd) doubles every round-trip time (RTT) until reaching the slow start threshold (ssthresh). When packet loss is detected via 3 duplicate ACKs, Fast Retransmit immediately resends the missing segment without waiting for a retransmission timeout timer."),
            ("4. Evolution of Web Protocols: HTTP/1.1, HTTP/2, and HTTP/3",
             "HTTP/1.1 introduced persistent connections and chunked transfer encoding, but suffered from Head-of-Line (HoL) blocking at the application level. HTTP/2 resolved this by introducing binary framing, header compression (HPACK), and multiplexed streams over a single TCP connection. However, TCP-level packet loss still stalls all multiplexed streams in HTTP/2. HTTP/3 eliminates TCP HoL blocking by running over QUIC, a transport protocol built on top of UDP. QUIC integrates TLS 1.3 encryption natively, reduces connection establishment latency to 0-RTT for repeated visits, and supports seamless connection migration across IP changes using Connection IDs.")
        ]
    ]
    build_pdf(doc3_path, "CS301: Computer Networks — Protocols, TCP/IP, and HTTP/3 QUIC", doc3_pages)

    # Document 4: Deep Learning & Transformers
    doc4_path = os.path.join(raw_dir, "cs401_deep_learning_and_transformers.pdf")
    doc4_pages = [
        # Page 1
        [
            ("1. The Transformer Architecture and Attention Mechanism",
             "Introduced in the seminal paper 'Attention Is All You Need' by Vaswani et al. (2017), the Transformer architecture replaced recurrent neural networks (RNNs) and LSTMs with attention mechanisms, enabling parallel training over long sequences. The core computational module is Scaled Dot-Product Attention, defined mathematically as: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V, where Q denotes the Queries matrix, K denotes the Keys matrix, V denotes the Values matrix, and d_k is the dimension of the keys."),
            ("2. Multi-Head Attention and Positional Encodings",
             "Multi-Head Attention projects Queries, Keys, and Values into h distinct linear subspaces, computes attention in parallel across each head, concatenates the outputs, and projects through a final linear layer. This allows the model to simultaneously attend to information from different representation subspaces at different positions. Because attention is permutation-invariant, Transformers inject Positional Encodings (sinusoidal functions or learnable embeddings like RoPE - Rotary Position Embedding) into input embeddings to preserve token order information.")
        ],
        # Page 2
        [
            ("3. Encoder-Decoder vs. Decoder-Only Models",
             "Transformers exist in three main configurations: 1) Encoder-Only models (e.g., BERT) use bidirectional self-attention to generate rich representations for classification and embeddings; 2) Encoder-Decoder models (e.g., T5, BART) process an input sequence with an encoder and autoregressively generate output sequences with a decoder, suitable for translation; 3) Decoder-Only models (e.g., GPT, Llama, Mistral) utilize causal masked self-attention where tokens can only attend to previous tokens, making them the standard architecture for modern Large Language Models (LLMs)."),
            ("4. Retrieval-Augmented Generation (RAG) Systems",
             "Retrieval-Augmented Generation (RAG) is a hybrid AI architecture that combines dense vector retrieval with autoregressive LLM generation. Instead of relying solely on the parametric knowledge encoded in model weights, RAG dynamically retrieves relevant external context from vector databases (such as ChromaDB) using semantic embeddings (such as Sentence Transformers). The retrieved context is inserted into the LLM system prompt, grounding the generation in verifiable facts, drastically reducing hallucinations, and enabling citations to original source documents.")
        ]
    ]
    build_pdf(doc4_path, "CS401: Deep Learning — Transformers, Attention, and RAG Architectures", doc4_pages)

    print("\nAll 4 educational CS PDF documents successfully generated in data/raw/!")

if __name__ == "__main__":
    main()
