from langchain_community.vectorstores import Chroma

db = Chroma(persist_directory="ck-6DEJFeP2LR8cAQYN6raUAqQzyfC6gRo31L7BVoxJaJzx")

print("Total documents:", db._collection.count())