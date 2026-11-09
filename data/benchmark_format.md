\# Data Format Description



The `benchmark\_sample\_50.json` file contains an array of objects. Each object represents a temporal mismatch pair. 



\- `project` (String): The name of the open-source Java project the code belongs to.

\- `historical\_comment` (String): The Javadoc comment block extracted from the historical version (unmodified).

\- `current\_code` (String): The current function body code (which may have undergone evolution).

\- `label` (String): The manual annotation, either `"DRIFT"` (semantic inconsistency) or `"NO\_DRIFT"` (semantics preserved).

