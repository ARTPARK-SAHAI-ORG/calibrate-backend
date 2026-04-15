import sqlite3
import json
import logging
import uuid
from os.path import join
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from routers.user_limits import UserLimits

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(join(os.getenv("DB_ROOT_DIR"), "pense.db"))

# Default user configuration — set via environment variables for local dev
DEFAULT_USER_EMAIL = os.getenv("DEFAULT_USER_EMAIL", "")
DEFAULT_USER_FIRST_NAME = os.getenv("DEFAULT_USER_FIRST_NAME", "")
DEFAULT_USER_LAST_NAME = os.getenv("DEFAULT_USER_LAST_NAME", "")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the database and create tables if they don't exist."""
    # Ensure the data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create users table first (other tables reference it)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'agent',
                config TEXT,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                UNIQUE(agent_id, tool_id),
                FOREIGN KEY (agent_id) REFERENCES agents(uuid),
                FOREIGN KEY (tool_id) REFERENCES tools(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                config TEXT,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                test_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                UNIQUE(agent_id, test_id),
                FOREIGN KEY (agent_id) REFERENCES agents(uuid),
                FOREIGN KEY (test_id) REFERENCES tests(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                user_id TEXT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                details TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_test_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                agent_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                details TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                simulation_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                details TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulation_id) REFERENCES simulations(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                agent_id TEXT DEFAULT NULL,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(uuid),
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                persona_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                UNIQUE(simulation_id, persona_id),
                FOREIGN KEY (simulation_id) REFERENCES simulations(uuid),
                FOREIGN KEY (persona_id) REFERENCES personas(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                UNIQUE(simulation_id, scenario_id),
                FOREIGN KEY (simulation_id) REFERENCES simulations(uuid),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                UNIQUE(simulation_id, metric_id),
                FOREIGN KEY (simulation_id) REFERENCES simulations(uuid),
                FOREIGN KEY (metric_id) REFERENCES metrics(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                dataset_id TEXT NOT NULL,
                audio_path TEXT DEFAULT NULL,
                text TEXT NOT NULL,
                order_index INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(uuid)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL UNIQUE,
                limits TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(uuid)
            )
        """
        )

        # Add deleted_at column to existing tables if not present (migration)
        tables_to_migrate = [
            "agents",
            "tools",
            "agent_tools",
            "tests",
            "agent_tests",
        ]
        for table in tables_to_migrate:
            try:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN deleted_at TIMESTAMP DEFAULT NULL"
                )
            except sqlite3.OperationalError:
                # Column already exists
                pass

        # Add agent_id column to simulations table if not present (migration)
        try:
            cursor.execute(
                "ALTER TABLE simulations ADD COLUMN agent_id TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Add password_hash column to users table (migration)
        try:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            pass

        # Add user_id column to all relevant tables if not present (migration)
        tables_with_user_id = [
            "agents",
            "tools",
            "tests",
            "personas",
            "scenarios",
            "metrics",
            "simulations",
            "jobs",
        ]
        for table in tables_with_user_id:
            try:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT NULL"
                )
            except sqlite3.OperationalError:
                # Column already exists
                pass

        try:
            cursor.execute(
                "ALTER TABLE agents ADD COLUMN type TEXT NOT NULL DEFAULT 'agent'"
            )
        except sqlite3.OperationalError:
            pass

        # Add is_public and share_token columns for public sharing feature
        for table in ("jobs", "agent_test_jobs", "simulation_jobs"):
            try:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN is_public INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN share_token TEXT DEFAULT NULL"
                )
            except sqlite3.OperationalError:
                pass

        conn.commit()

        # Create default user if not exists and update existing rows with NULL user_id
        cursor.execute("SELECT uuid FROM users WHERE email = ?", (DEFAULT_USER_EMAIL,))
        default_user_row = cursor.fetchone()

        if default_user_row:
            default_user_uuid = default_user_row["uuid"]
            logger.info(f"Default user already exists with UUID: {default_user_uuid}")
        else:
            # Create the default user
            default_user_uuid = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO users (uuid, first_name, last_name, email)
                VALUES (?, ?, ?, ?)
                """,
                (
                    default_user_uuid,
                    DEFAULT_USER_FIRST_NAME,
                    DEFAULT_USER_LAST_NAME,
                    DEFAULT_USER_EMAIL,
                ),
            )
            conn.commit()
            logger.info(f"Created default user with UUID: {default_user_uuid}")

        # Update all existing rows with NULL user_id to use the default user
        for table in tables_with_user_id:
            cursor.execute(
                f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL",
                (default_user_uuid,),
            )
            rows_updated = cursor.rowcount
            if rows_updated > 0:
                logger.info(
                    f"Updated {rows_updated} row(s) in {table} with default user_id"
                )

        conn.commit()
        logger.info("Database initialized successfully")


# ============ Users Functions ============


def create_user(
    first_name: str,
    last_name: str,
    email: str,
) -> str:
    """Create a new user and return its UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        user_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO users (uuid, first_name, last_name, email)
            VALUES (?, ?, ?, ?)
            """,
            (user_uuid, first_name, last_name, email),
        )
        conn.commit()
        logger.info(f"Created user with UUID: {user_uuid}")
        return user_uuid


def get_user(user_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a user by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE uuid = ?", (user_uuid,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get a user by email."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_users() -> List[Dict[str, Any]]:
    """Get all users."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_user(
    user_uuid: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    email: Optional[str] = None,
) -> bool:
    """Update a user. Returns True if the user was found and updated."""
    updates = []
    params = []

    if first_name is not None:
        updates.append("first_name = ?")
        params.append(first_name)
    if last_name is not None:
        updates.append("last_name = ?")
        params.append(last_name)
    if email is not None:
        updates.append("email = ?")
        params.append(email)

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(user_uuid)

    query = f"UPDATE users SET {', '.join(updates)} WHERE uuid = ?"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated user with UUID: {user_uuid}")
        return updated


def delete_user(user_uuid: str) -> bool:
    """Delete a user. Returns True if the user was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE uuid = ?", (user_uuid,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted user with UUID: {user_uuid}")
        return deleted


def get_or_create_user(
    email: str,
    first_name: str,
    last_name: str,
) -> Dict[str, Any]:
    """Get a user by email, or create a new one if not found."""
    user = get_user_by_email(email)
    if user:
        # Update name if changed
        if user["first_name"] != first_name or user["last_name"] != last_name:
            update_user(user["uuid"], first_name=first_name, last_name=last_name)
            user = get_user(user["uuid"])
        return user

    # Create new user
    user_uuid = create_user(first_name=first_name, last_name=last_name, email=email)
    return get_user(user_uuid)


def create_user_with_password(
    first_name: str,
    last_name: str,
    email: str,
    password_hash: str,
) -> str:
    """Create a new user with email/password and return its UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        user_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO users (uuid, first_name, last_name, email, password_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_uuid, first_name, last_name, email, password_hash),
        )
        conn.commit()
        logger.info(f"Created user (email/password auth) with UUID: {user_uuid}")
        return user_uuid


# ============ Agents Functions ============


def create_agent(
    name: str,
    agent_type: str = "agent",
    config: Optional[Dict[str, Any]] = None,
    user_id: str = None,
) -> str:
    """Create a new agent and return its UUID.

    Args:
        name: Name of the agent
        agent_type: Type of agent — 'agent' or 'connection'
        config: Optional configuration dict
        user_id: UUID of the user creating this agent (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating an agent")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        agent_uuid = str(uuid.uuid4())
        config_json = json.dumps(config) if config is not None else None
        cursor.execute(
            """
            INSERT INTO agents (uuid, name, type, config, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agent_uuid, name, agent_type, config_json, user_id),
        )
        conn.commit()
        logger.info(f"Created agent with UUID: {agent_uuid}")
        return agent_uuid


def _parse_agent_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a database row and deserialize JSON fields."""
    agent = dict(row)
    # Deserialize config from JSON string
    if agent.get("config"):
        agent["config"] = json.loads(agent["config"])

    return agent


def get_agent(agent_uuid: str) -> Optional[Dict[str, Any]]:
    """Get an agent by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agents WHERE uuid = ? AND deleted_at IS NULL", (agent_uuid,)
        )
        row = cursor.fetchone()
        if row:
            return _parse_agent_row(row)
        return None


def get_all_agents(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all agents, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM agents WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM agents WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [_parse_agent_row(row) for row in rows]


def update_agent(
    agent_uuid: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update an agent. Returns True if the agent was found and updated."""
    # Build dynamic update query
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if config is not None:
        updates.append("config = ?")
        # Serialize config to JSON string for storage
        params.append(json.dumps(config))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(agent_uuid)

    query = (
        f"UPDATE agents SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"
    )

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated agent with UUID: {agent_uuid}")
        return updated


def delete_agent(agent_uuid: str) -> bool:
    """Soft delete an agent. Returns True if the agent was found and deleted.
    Also soft deletes related agent_tools and agent_tests.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE agents SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (agent_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            # Soft delete related agent_tools
            cursor.execute(
                "UPDATE agent_tools SET deleted_at = CURRENT_TIMESTAMP WHERE agent_id = ? AND deleted_at IS NULL",
                (agent_uuid,),
            )
            # Soft delete related agent_tests
            cursor.execute(
                "UPDATE agent_tests SET deleted_at = CURRENT_TIMESTAMP WHERE agent_id = ? AND deleted_at IS NULL",
                (agent_uuid,),
            )
            logger.info(f"Soft deleted agent with UUID: {agent_uuid}")

        conn.commit()
        return deleted


def create_tool(
    name: str,
    description: str,
    config: Optional[Dict[str, Any]] = None,
    user_id: str = None,
) -> str:
    """Create a new tool and return its UUID.

    Args:
        name: Name of the tool
        description: Description of the tool
        config: Optional configuration dict
        user_id: UUID of the user creating this tool (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating a tool")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Generate UUID for the tool
        tool_uuid = str(uuid.uuid4())
        # Serialize config to JSON string for storage
        config_json = json.dumps(config) if config is not None else None
        cursor.execute(
            """
            INSERT INTO tools (uuid, name, description, config, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tool_uuid, name, description, config_json, user_id),
        )
        conn.commit()
        logger.info(f"Created tool with UUID: {tool_uuid}")
        return tool_uuid


def _parse_tool_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a database row and deserialize JSON fields."""
    tool = dict(row)
    # Deserialize config from JSON string
    if tool.get("config"):
        tool["config"] = json.loads(tool["config"])

    return tool


def get_tool(tool_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a tool by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tools WHERE uuid = ? AND deleted_at IS NULL", (tool_uuid,)
        )
        row = cursor.fetchone()
        if row:
            return _parse_tool_row(row)
        return None


def get_all_tools(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all tools, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM tools WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM tools WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [_parse_tool_row(row) for row in rows]


def update_tool(
    tool_uuid: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a tool. Returns True if the tool was found and updated."""
    # Build dynamic update query
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if config is not None:
        updates.append("config = ?")
        # Serialize config to JSON string for storage
        params.append(json.dumps(config))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(tool_uuid)

    query = (
        f"UPDATE tools SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"
    )

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated tool with UUID: {tool_uuid}")
        return updated


def delete_tool(tool_uuid: str) -> bool:
    """Soft delete a tool. Returns True if the tool was found and deleted.
    Also soft deletes related agent_tools entries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tools SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (tool_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            # Soft delete related agent_tools
            cursor.execute(
                "UPDATE agent_tools SET deleted_at = CURRENT_TIMESTAMP WHERE tool_id = ? AND deleted_at IS NULL",
                (tool_uuid,),
            )
            logger.info(f"Soft deleted tool with UUID: {tool_uuid}")

        conn.commit()
        return deleted


def add_tool_to_agent(agent_id: str, tool_id: str) -> int:
    """Add a tool to an agent. Returns the id of the created/restored link.
    If a soft-deleted link exists, it will be restored by unsetting deleted_at.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if a soft-deleted link exists
        cursor.execute(
            "SELECT id FROM agent_tools WHERE agent_id = ? AND tool_id = ? AND deleted_at IS NOT NULL",
            (agent_id, tool_id),
        )
        existing = cursor.fetchone()
        if existing:
            # Restore the soft-deleted link
            cursor.execute(
                "UPDATE agent_tools SET deleted_at = NULL WHERE id = ?",
                (existing["id"],),
            )
            conn.commit()
            logger.info(f"Restored tool {tool_id} to agent {agent_id}")
            return existing["id"]

        # Insert new link
        cursor.execute(
            """
            INSERT INTO agent_tools (agent_id, tool_id)
            VALUES (?, ?)
            """,
            (agent_id, tool_id),
        )
        conn.commit()
        link_id = cursor.lastrowid
        logger.info(f"Added tool {tool_id} to agent {agent_id}")
        return link_id


def remove_tool_from_agent(agent_id: str, tool_id: str) -> bool:
    """Soft delete a tool from an agent. Returns True if the link was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE agent_tools SET deleted_at = CURRENT_TIMESTAMP WHERE agent_id = ? AND tool_id = ? AND deleted_at IS NULL",
            (agent_id, tool_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Soft deleted tool {tool_id} from agent {agent_id}")

        return deleted


def get_tools_for_agent(agent_id: str) -> List[Dict[str, Any]]:
    """Get all tools associated with an agent."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.* FROM tools t
            INNER JOIN agent_tools at ON t.uuid = at.tool_id
            WHERE at.agent_id = ? AND at.deleted_at IS NULL AND t.deleted_at IS NULL
            ORDER BY at.created_at DESC
            """,
            (agent_id,),
        )
        rows = cursor.fetchall()
        return [_parse_tool_row(row) for row in rows]


def get_agents_for_tool(tool_id: str) -> List[Dict[str, Any]]:
    """Get all agents associated with a tool."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.* FROM agents a
            INNER JOIN agent_tools at ON a.uuid = at.agent_id
            WHERE at.tool_id = ? AND at.deleted_at IS NULL AND a.deleted_at IS NULL
            ORDER BY at.created_at DESC
            """,
            (tool_id,),
        )
        rows = cursor.fetchall()
        return [_parse_agent_row(row) for row in rows]


def get_agent_tool_link(agent_id: str, tool_id: str) -> Optional[Dict[str, Any]]:
    """Check if a specific agent-tool link exists."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_tools WHERE agent_id = ? AND tool_id = ? AND deleted_at IS NULL",
            (agent_id, tool_id),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_agent_tools() -> List[Dict[str, Any]]:
    """Get all agent-tool links."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_tools WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============ Tests Functions ============


def create_test(
    name: str,
    type: str,
    config: Optional[Dict[str, Any]] = None,
    user_id: str = None,
) -> str:
    """Create a new test and return its UUID.

    Args:
        name: Name of the test
        type: Type of the test
        config: Optional configuration dict
        user_id: UUID of the user creating this test (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating a test")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        test_uuid = str(uuid.uuid4())
        config_json = json.dumps(config) if config is not None else None
        cursor.execute(
            """
            INSERT INTO tests (uuid, name, type, config, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (test_uuid, name, type, config_json, user_id),
        )
        conn.commit()
        logger.info(f"Created test with UUID: {test_uuid}")
        return test_uuid


def bulk_create_tests(
    tests: List[Dict[str, Any]],
    user_id: str,
) -> List[str]:
    """Create multiple tests in a single transaction and return their UUIDs.

    Each item in tests must have keys: name, type, config.
    Raises ValueError if user_id is missing or any name collides with an
    existing (non-deleted) test owned by the same user.
    """
    if not user_id:
        raise ValueError("user_id is required when creating tests")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        names = [t["name"] for t in tests]
        placeholders = ",".join("?" for _ in names)
        cursor.execute(
            f"SELECT name FROM tests WHERE user_id = ? AND deleted_at IS NULL AND name IN ({placeholders})",
            [user_id] + names,
        )
        existing = {row["name"] for row in cursor.fetchall()}
        if existing:
            raise ValueError(f"Test names already exist: {', '.join(sorted(existing))}")

        uuids: List[str] = []
        for t in tests:
            test_uuid = str(uuid.uuid4())
            config_json = (
                json.dumps(t["config"]) if t.get("config") is not None else None
            )
            cursor.execute(
                """
                INSERT INTO tests (uuid, name, type, config, user_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (test_uuid, t["name"], t["type"], config_json, user_id),
            )
            uuids.append(test_uuid)

        conn.commit()
        logger.info(f"Bulk created {len(uuids)} tests")
        return uuids


def _parse_test_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a database row and deserialize JSON fields."""
    test = dict(row)
    if test.get("config"):
        test["config"] = json.loads(test["config"])
    return test


def get_test(test_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a test by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tests WHERE uuid = ? AND deleted_at IS NULL", (test_uuid,)
        )
        row = cursor.fetchone()
        if row:
            return _parse_test_row(row)
        return None


def get_all_tests(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all tests, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM tests WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM tests WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [_parse_test_row(row) for row in rows]


def update_test(
    test_uuid: str,
    name: Optional[str] = None,
    type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a test. Returns True if the test was found and updated."""
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if type is not None:
        updates.append("type = ?")
        params.append(type)
    if config is not None:
        updates.append("config = ?")
        params.append(json.dumps(config))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(test_uuid)

    query = (
        f"UPDATE tests SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"
    )

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated test with UUID: {test_uuid}")
        return updated


def delete_test(test_uuid: str) -> bool:
    """Soft delete a test. Returns True if the test was found and deleted.
    Also soft deletes related agent_tests entries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tests SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (test_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            # Soft delete related agent_tests
            cursor.execute(
                "UPDATE agent_tests SET deleted_at = CURRENT_TIMESTAMP WHERE test_id = ? AND deleted_at IS NULL",
                (test_uuid,),
            )
            logger.info(f"Soft deleted test with UUID: {test_uuid}")

        conn.commit()
        return deleted


def bulk_delete_tests(test_uuids: List[str], user_id: str) -> int:
    """Soft delete multiple tests owned by user_id.
    Also soft deletes related agent_tests entries.
    Returns the number of tests actually deleted.
    """
    if not test_uuids:
        return 0

    placeholders = ",".join("?" for _ in test_uuids)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE tests SET deleted_at = CURRENT_TIMESTAMP "
            f"WHERE uuid IN ({placeholders}) AND user_id = ? AND deleted_at IS NULL",
            (*test_uuids, user_id),
        )
        deleted_count = cursor.rowcount

        if deleted_count > 0:
            cursor.execute(
                f"UPDATE agent_tests SET deleted_at = CURRENT_TIMESTAMP "
                f"WHERE test_id IN ({placeholders}) AND deleted_at IS NULL",
                test_uuids,
            )
            logger.info(f"Bulk soft deleted {deleted_count} tests for user {user_id}")

        conn.commit()
        return deleted_count


# ============ Personas Functions ============


def create_persona(
    name: str,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    user_id: str = None,
) -> str:
    """Create a new persona and return its UUID.

    Args:
        name: Name of the persona
        description: Optional description
        config: Optional configuration dict
        user_id: UUID of the user creating this persona (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating a persona")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        persona_uuid = str(uuid.uuid4())
        config_json = json.dumps(config) if config is not None else None
        cursor.execute(
            """
            INSERT INTO personas (uuid, name, description, config, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (persona_uuid, name, description, config_json, user_id),
        )
        conn.commit()
        logger.info(f"Created persona with UUID: {persona_uuid}")
        return persona_uuid


def _parse_persona_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a persona database row and deserialize JSON fields."""
    persona = dict(row)
    if persona.get("config"):
        persona["config"] = json.loads(persona["config"])
    return persona


def get_persona(persona_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a persona by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM personas WHERE uuid = ? AND deleted_at IS NULL",
            (persona_uuid,),
        )
        row = cursor.fetchone()
        if row:
            return _parse_persona_row(row)
        return None


def get_all_personas(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all personas, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM personas WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM personas WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [_parse_persona_row(row) for row in rows]


def update_persona(
    persona_uuid: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a persona. Returns True if the persona was found and updated."""
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if config is not None:
        updates.append("config = ?")
        params.append(json.dumps(config))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(persona_uuid)

    query = f"UPDATE personas SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated persona with UUID: {persona_uuid}")
        return updated


def delete_persona(persona_uuid: str) -> bool:
    """Soft delete a persona. Returns True if the persona was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE personas SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (persona_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Soft deleted persona with UUID: {persona_uuid}")

        conn.commit()
        return deleted


# ============ Scenarios Functions ============


def create_scenario(
    name: str,
    description: Optional[str] = None,
    user_id: str = None,
) -> str:
    """Create a new scenario and return its UUID.

    Args:
        name: Name of the scenario
        description: Optional description
        user_id: UUID of the user creating this scenario (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating a scenario")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        scenario_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO scenarios (uuid, name, description, user_id)
            VALUES (?, ?, ?, ?)
            """,
            (scenario_uuid, name, description, user_id),
        )
        conn.commit()
        logger.info(f"Created scenario with UUID: {scenario_uuid}")
        return scenario_uuid


def get_scenario(scenario_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a scenario by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM scenarios WHERE uuid = ? AND deleted_at IS NULL",
            (scenario_uuid,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_scenarios(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all scenarios, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM scenarios WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM scenarios WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_scenario(
    scenario_uuid: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """Update a scenario. Returns True if the scenario was found and updated."""
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(scenario_uuid)

    query = f"UPDATE scenarios SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated scenario with UUID: {scenario_uuid}")
        return updated


def delete_scenario(scenario_uuid: str) -> bool:
    """Soft delete a scenario. Returns True if the scenario was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE scenarios SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (scenario_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Soft deleted scenario with UUID: {scenario_uuid}")

        conn.commit()
        return deleted


# ============ Metrics Functions ============


def create_metric(
    name: str,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    user_id: str = None,
) -> str:
    """Create a new metric and return its UUID.

    Args:
        name: Name of the metric
        description: Optional description
        config: Optional configuration dict
        user_id: UUID of the user creating this metric (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating a metric")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        metric_uuid = str(uuid.uuid4())
        config_json = json.dumps(config) if config is not None else None
        cursor.execute(
            """
            INSERT INTO metrics (uuid, name, description, config, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (metric_uuid, name, description, config_json, user_id),
        )
        conn.commit()
        logger.info(f"Created metric with UUID: {metric_uuid}")
        return metric_uuid


def _parse_metric_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a metric database row and deserialize JSON fields."""
    metric = dict(row)
    if metric.get("config"):
        metric["config"] = json.loads(metric["config"])
    return metric


def get_metric(metric_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a metric by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM metrics WHERE uuid = ? AND deleted_at IS NULL",
            (metric_uuid,),
        )
        row = cursor.fetchone()
        if row:
            return _parse_metric_row(row)
        return None


def get_all_metrics(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all metrics, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM metrics WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM metrics WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [_parse_metric_row(row) for row in rows]


def update_metric(
    metric_uuid: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a metric. Returns True if the metric was found and updated."""
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if config is not None:
        updates.append("config = ?")
        params.append(json.dumps(config))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(metric_uuid)

    query = (
        f"UPDATE metrics SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"
    )

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated metric with UUID: {metric_uuid}")
        return updated


def delete_metric(metric_uuid: str) -> bool:
    """Soft delete a metric. Returns True if the metric was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE metrics SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (metric_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Soft deleted metric with UUID: {metric_uuid}")

        conn.commit()
        return deleted


# ============ Simulations Functions ============


def create_simulation(
    name: str, agent_id: Optional[str] = None, user_id: str = None
) -> str:
    """Create a new simulation and return its UUID.

    Args:
        name: Name of the simulation
        agent_id: Optional UUID of the linked agent
        user_id: UUID of the user creating this simulation (required)

    Raises:
        ValueError: If user_id is not provided
    """
    if not user_id:
        raise ValueError("user_id is required when creating a simulation")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        simulation_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO simulations (uuid, name, agent_id, user_id)
            VALUES (?, ?, ?, ?)
            """,
            (simulation_uuid, name, agent_id, user_id),
        )
        conn.commit()
        logger.info(f"Created simulation with UUID: {simulation_uuid}")
        return simulation_uuid


def get_simulation(simulation_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a simulation by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulations WHERE uuid = ? AND deleted_at IS NULL",
            (simulation_uuid,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_simulations(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all simulations, optionally filtered by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM simulations WHERE deleted_at IS NULL AND user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT * FROM simulations WHERE deleted_at IS NULL ORDER BY created_at DESC"
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_simulation(
    simulation_uuid: str,
    name: Optional[str] = None,
    agent_id: Optional[str] = None,
    clear_agent: bool = False,
) -> bool:
    """Update a simulation. Returns True if the simulation was found and updated.

    Args:
        simulation_uuid: UUID of the simulation to update
        name: New name for the simulation
        agent_id: New agent ID to link to the simulation
        clear_agent: If True, clears the agent_id (sets to NULL)
    """
    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)

    if clear_agent:
        updates.append("agent_id = NULL")
    elif agent_id is not None:
        updates.append("agent_id = ?")
        params.append(agent_id)

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(simulation_uuid)

    query = f"UPDATE simulations SET {', '.join(updates)} WHERE uuid = ? AND deleted_at IS NULL"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated simulation with UUID: {simulation_uuid}")
        return updated


def delete_simulation(simulation_uuid: str) -> bool:
    """Soft delete a simulation. Returns True if the simulation was found and deleted.
    Also soft deletes related simulation_personas, simulation_scenarios, and simulation_metrics entries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE simulations SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND deleted_at IS NULL",
            (simulation_uuid,),
        )
        deleted = cursor.rowcount > 0

        if deleted:
            # Soft delete related pivot table entries
            cursor.execute(
                "UPDATE simulation_personas SET deleted_at = CURRENT_TIMESTAMP WHERE simulation_id = ? AND deleted_at IS NULL",
                (simulation_uuid,),
            )
            cursor.execute(
                "UPDATE simulation_scenarios SET deleted_at = CURRENT_TIMESTAMP WHERE simulation_id = ? AND deleted_at IS NULL",
                (simulation_uuid,),
            )
            cursor.execute(
                "UPDATE simulation_metrics SET deleted_at = CURRENT_TIMESTAMP WHERE simulation_id = ? AND deleted_at IS NULL",
                (simulation_uuid,),
            )
            logger.info(f"Soft deleted simulation with UUID: {simulation_uuid}")

        conn.commit()
        return deleted


# ============ Simulation Personas Functions ============


def add_persona_to_simulation(simulation_id: str, persona_id: str) -> int:
    """Add a persona to a simulation. Returns the id of the created/restored link."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if a soft-deleted link exists
        cursor.execute(
            "SELECT id FROM simulation_personas WHERE simulation_id = ? AND persona_id = ? AND deleted_at IS NOT NULL",
            (simulation_id, persona_id),
        )
        existing = cursor.fetchone()
        if existing:
            # Restore the soft-deleted link
            cursor.execute(
                "UPDATE simulation_personas SET deleted_at = NULL WHERE id = ?",
                (existing["id"],),
            )
            conn.commit()
            logger.info(f"Restored persona {persona_id} to simulation {simulation_id}")
            return existing["id"]

        # Insert new link
        cursor.execute(
            """
            INSERT INTO simulation_personas (simulation_id, persona_id)
            VALUES (?, ?)
            """,
            (simulation_id, persona_id),
        )
        conn.commit()
        link_id = cursor.lastrowid
        logger.info(f"Added persona {persona_id} to simulation {simulation_id}")
        return link_id


def remove_persona_from_simulation(simulation_id: str, persona_id: str) -> bool:
    """Soft delete a persona from a simulation. Returns True if the link was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE simulation_personas SET deleted_at = CURRENT_TIMESTAMP WHERE simulation_id = ? AND persona_id = ? AND deleted_at IS NULL",
            (simulation_id, persona_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Removed persona {persona_id} from simulation {simulation_id}")
        return deleted


def get_personas_for_simulation(simulation_id: str) -> List[Dict[str, Any]]:
    """Get all personas for a simulation."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.* FROM personas p
            INNER JOIN simulation_personas sp ON p.uuid = sp.persona_id
            WHERE sp.simulation_id = ? AND sp.deleted_at IS NULL AND p.deleted_at IS NULL
            ORDER BY sp.created_at DESC
            """,
            (simulation_id,),
        )
        rows = cursor.fetchall()
        return [_parse_persona_row(row) for row in rows]


def get_simulation_persona_link(
    simulation_id: str, persona_id: str
) -> Optional[Dict[str, Any]]:
    """Get a specific simulation-persona link."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_personas WHERE simulation_id = ? AND persona_id = ? AND deleted_at IS NULL",
            (simulation_id, persona_id),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_simulation_personas() -> List[Dict[str, Any]]:
    """Get all simulation-persona links."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_personas WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============ Simulation Scenarios Functions ============


def add_scenario_to_simulation(simulation_id: str, scenario_id: str) -> int:
    """Add a scenario to a simulation. Returns the id of the created/restored link."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if a soft-deleted link exists
        cursor.execute(
            "SELECT id FROM simulation_scenarios WHERE simulation_id = ? AND scenario_id = ? AND deleted_at IS NOT NULL",
            (simulation_id, scenario_id),
        )
        existing = cursor.fetchone()
        if existing:
            # Restore the soft-deleted link
            cursor.execute(
                "UPDATE simulation_scenarios SET deleted_at = NULL WHERE id = ?",
                (existing["id"],),
            )
            conn.commit()
            logger.info(
                f"Restored scenario {scenario_id} to simulation {simulation_id}"
            )
            return existing["id"]

        # Insert new link
        cursor.execute(
            """
            INSERT INTO simulation_scenarios (simulation_id, scenario_id)
            VALUES (?, ?)
            """,
            (simulation_id, scenario_id),
        )
        conn.commit()
        link_id = cursor.lastrowid
        logger.info(f"Added scenario {scenario_id} to simulation {simulation_id}")
        return link_id


def remove_scenario_from_simulation(simulation_id: str, scenario_id: str) -> bool:
    """Soft delete a scenario from a simulation. Returns True if the link was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE simulation_scenarios SET deleted_at = CURRENT_TIMESTAMP WHERE simulation_id = ? AND scenario_id = ? AND deleted_at IS NULL",
            (simulation_id, scenario_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(
                f"Removed scenario {scenario_id} from simulation {simulation_id}"
            )
        return deleted


def get_scenarios_for_simulation(simulation_id: str) -> List[Dict[str, Any]]:
    """Get all scenarios for a simulation."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.* FROM scenarios s
            INNER JOIN simulation_scenarios ss ON s.uuid = ss.scenario_id
            WHERE ss.simulation_id = ? AND ss.deleted_at IS NULL AND s.deleted_at IS NULL
            ORDER BY ss.created_at DESC
            """,
            (simulation_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_simulation_scenario_link(
    simulation_id: str, scenario_id: str
) -> Optional[Dict[str, Any]]:
    """Get a specific simulation-scenario link."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_scenarios WHERE simulation_id = ? AND scenario_id = ? AND deleted_at IS NULL",
            (simulation_id, scenario_id),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_simulation_scenarios() -> List[Dict[str, Any]]:
    """Get all simulation-scenario links."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_scenarios WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============ Simulation Metrics Functions ============


def add_metric_to_simulation(simulation_id: str, metric_id: str) -> int:
    """Add a metric to a simulation. Returns the id of the created/restored link."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if a soft-deleted link exists
        cursor.execute(
            "SELECT id FROM simulation_metrics WHERE simulation_id = ? AND metric_id = ? AND deleted_at IS NOT NULL",
            (simulation_id, metric_id),
        )
        existing = cursor.fetchone()
        if existing:
            # Restore the soft-deleted link
            cursor.execute(
                "UPDATE simulation_metrics SET deleted_at = NULL WHERE id = ?",
                (existing["id"],),
            )
            conn.commit()
            logger.info(f"Restored metric {metric_id} to simulation {simulation_id}")
            return existing["id"]

        # Insert new link
        cursor.execute(
            """
            INSERT INTO simulation_metrics (simulation_id, metric_id)
            VALUES (?, ?)
            """,
            (simulation_id, metric_id),
        )
        conn.commit()
        link_id = cursor.lastrowid
        logger.info(f"Added metric {metric_id} to simulation {simulation_id}")
        return link_id


def remove_metric_from_simulation(simulation_id: str, metric_id: str) -> bool:
    """Soft delete a metric from a simulation. Returns True if the link was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE simulation_metrics SET deleted_at = CURRENT_TIMESTAMP WHERE simulation_id = ? AND metric_id = ? AND deleted_at IS NULL",
            (simulation_id, metric_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Removed metric {metric_id} from simulation {simulation_id}")
        return deleted


def get_metrics_for_simulation(simulation_id: str) -> List[Dict[str, Any]]:
    """Get all metrics for a simulation."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT m.* FROM metrics m
            INNER JOIN simulation_metrics sm ON m.uuid = sm.metric_id
            WHERE sm.simulation_id = ? AND sm.deleted_at IS NULL AND m.deleted_at IS NULL
            ORDER BY sm.created_at DESC
            """,
            (simulation_id,),
        )
        rows = cursor.fetchall()
        return [_parse_metric_row(row) for row in rows]


def get_simulation_metric_link(
    simulation_id: str, metric_id: str
) -> Optional[Dict[str, Any]]:
    """Get a specific simulation-metric link."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_metrics WHERE simulation_id = ? AND metric_id = ? AND deleted_at IS NULL",
            (simulation_id, metric_id),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_simulation_metrics() -> List[Dict[str, Any]]:
    """Get all simulation-metric links."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_metrics WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============ Agent Tests Functions ============


def add_test_to_agent(agent_id: str, test_id: str) -> int:
    """Add a test to an agent. Returns the id of the created/restored link.
    If a soft-deleted link exists, it will be restored by unsetting deleted_at.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if a soft-deleted link exists
        cursor.execute(
            "SELECT id FROM agent_tests WHERE agent_id = ? AND test_id = ? AND deleted_at IS NOT NULL",
            (agent_id, test_id),
        )
        existing = cursor.fetchone()
        if existing:
            # Restore the soft-deleted link
            cursor.execute(
                "UPDATE agent_tests SET deleted_at = NULL WHERE id = ?",
                (existing["id"],),
            )
            conn.commit()
            logger.info(f"Restored test {test_id} to agent {agent_id}")
            return existing["id"]
        else:
            # Insert new link
            cursor.execute(
                """
                INSERT INTO agent_tests (agent_id, test_id)
                VALUES (?, ?)
                """,
                (agent_id, test_id),
            )
            conn.commit()
            link_id = cursor.lastrowid
            logger.info(f"Added test {test_id} to agent {agent_id}")
            return link_id


def remove_test_from_agent(agent_id: str, test_id: str) -> bool:
    """Soft delete a test from an agent. Returns True if the link was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE agent_tests SET deleted_at = CURRENT_TIMESTAMP WHERE agent_id = ? AND test_id = ? AND deleted_at IS NULL",
            (agent_id, test_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Soft deleted test {test_id} from agent {agent_id}")

        return deleted


def bulk_remove_tests_from_agent(agent_id: str, test_ids: List[str]) -> int:
    """Soft delete multiple test links from an agent. Returns the number of links removed."""
    if not test_ids:
        return 0

    placeholders = ",".join("?" for _ in test_ids)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE agent_tests SET deleted_at = CURRENT_TIMESTAMP "
            f"WHERE agent_id = ? AND test_id IN ({placeholders}) AND deleted_at IS NULL",
            (agent_id, *test_ids),
        )
        conn.commit()
        deleted_count = cursor.rowcount

        if deleted_count > 0:
            logger.info(
                f"Bulk soft deleted {deleted_count} test links from agent {agent_id}"
            )

        return deleted_count


def get_tests_for_agent(agent_id: str) -> List[Dict[str, Any]]:
    """Get all tests associated with an agent."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.* FROM tests t
            INNER JOIN agent_tests at ON t.uuid = at.test_id
            WHERE at.agent_id = ? AND at.deleted_at IS NULL AND t.deleted_at IS NULL
            ORDER BY at.created_at DESC
            """,
            (agent_id,),
        )
        rows = cursor.fetchall()
        return [_parse_test_row(row) for row in rows]


def get_agents_for_test(test_id: str) -> List[Dict[str, Any]]:
    """Get all agents associated with a test."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.* FROM agents a
            INNER JOIN agent_tests at ON a.uuid = at.agent_id
            WHERE at.test_id = ? AND at.deleted_at IS NULL AND a.deleted_at IS NULL
            ORDER BY at.created_at DESC
            """,
            (test_id,),
        )
        rows = cursor.fetchall()
        return [_parse_agent_row(row) for row in rows]


def get_agent_test_link(agent_id: str, test_id: str) -> Optional[Dict[str, Any]]:
    """Check if a specific agent-test link exists."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_tests WHERE agent_id = ? AND test_id = ? AND deleted_at IS NULL",
            (agent_id, test_id),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_agent_tests() -> List[Dict[str, Any]]:
    """Get all agent-test links."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_tests WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# ============ Jobs Functions ============


def create_job(
    job_type: str,
    user_id: str,
    status: str = "in_progress",
    details: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new job and return its UUID.

    Args:
        job_type: Type of job (stt-eval, tts-eval, llm-unit-test, llm-benchmark)
        user_id: UUID of the user who owns this job
        status: Initial status (defaults to 'in_progress')
        details: JSON config needed to re-trigger the job if interrupted
        results: Initial results (usually None)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        job_uuid = str(uuid.uuid4())
        details_json = json.dumps(details) if details is not None else None
        results_json = json.dumps(results) if results is not None else None
        cursor.execute(
            """
            INSERT INTO jobs (uuid, user_id, type, status, details, results)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_uuid, user_id, job_type, status, details_json, results_json),
        )
        conn.commit()
        logger.info(
            f"Created job with UUID: {job_uuid}, type: {job_type}, user_id: {user_id}"
        )
        return job_uuid


def _parse_job_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a job database row and deserialize JSON fields."""
    job = dict(row)
    if job.get("details"):
        job["details"] = json.loads(job["details"])
    if job.get("results"):
        job["results"] = json.loads(job["results"])
    return job


def get_job(job_uuid: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a job by UUID, optionally filtered by user_id.

    Args:
        job_uuid: The UUID of the job
        user_id: If provided, only return the job if it belongs to this user
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM jobs WHERE uuid = ? AND user_id = ?",
                (job_uuid, user_id),
            )
        else:
            cursor.execute("SELECT * FROM jobs WHERE uuid = ?", (job_uuid,))
        row = cursor.fetchone()
        if row:
            return _parse_job_row(row)
        return None


def get_all_jobs(user_id: str, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all jobs for a user, optionally filtered by type.

    Args:
        user_id: UUID of the user who owns the jobs
        job_type: Optional filter by job type
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM jobs WHERE user_id = ? AND type = ? ORDER BY created_at DESC",
                (user_id, job_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        rows = cursor.fetchall()
        return [_parse_job_row(row) for row in rows]


def get_pending_jobs() -> List[Dict[str, Any]]:
    """Get all jobs with status 'in_progress' (for recovery on restart)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM jobs WHERE status = 'in_progress' ORDER BY created_at ASC"
        )
        rows = cursor.fetchall()
        return [_parse_job_row(row) for row in rows]


def get_queued_jobs(job_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get all jobs with status 'queued', optionally filtered by job types."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"SELECT * FROM jobs WHERE status = 'queued' AND type IN ({placeholders}) ORDER BY created_at ASC",
                job_types,
            )
        else:
            cursor.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC"
            )
        rows = cursor.fetchall()
        return [_parse_job_row(row) for row in rows]


def count_running_jobs(job_types: Optional[List[str]] = None) -> int:
    """Count jobs with status 'in_progress', optionally filtered by job types."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status = 'in_progress' AND type IN ({placeholders})",
                job_types,
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'in_progress'")
        return cursor.fetchone()[0]


def count_running_jobs_for_user(
    user_id: str, job_types: Optional[List[str]] = None
) -> int:
    """Count jobs with status 'in_progress' for a specific user, optionally filtered by job types."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status = 'in_progress' AND user_id = ? AND type IN ({placeholders})",
                [user_id] + job_types,
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'in_progress' AND user_id = ?",
                (user_id,),
            )
        return cursor.fetchone()[0]


def update_job(
    job_uuid: str,
    status: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a job. Returns True if the job was found and updated.

    If details is provided, it will be merged with existing details (not replaced).
    """
    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if results is not None:
        updates.append("results = ?")
        params.append(json.dumps(results))

    # For details, we need to merge with existing details
    if details is not None:
        # First, fetch existing details
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT details FROM jobs WHERE uuid = ?", (job_uuid,))
            row = cursor.fetchone()
            if row and row[0]:
                existing_details = json.loads(row[0])
                # Merge new details into existing
                existing_details.update(details)
                details = existing_details
        updates.append("details = ?")
        params.append(json.dumps(details))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(job_uuid)

    query = f"UPDATE jobs SET {', '.join(updates)} WHERE uuid = ?"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated job with UUID: {job_uuid}")
        return updated


def update_job_visibility(
    job_uuid: str, is_public: bool, share_token: Optional[str]
) -> bool:
    """Update is_public and share_token for a job. Returns True if the job was found."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET is_public = ?, share_token = ?, updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
            (1 if is_public else 0, share_token, job_uuid),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_job_by_share_token(
    share_token: str, job_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get a job by its share_token, optionally restricted to a specific job type.

    Always filters to is_public = 1. Pass job_type (e.g. 'stt-eval', 'tts-eval')
    to prevent tokens from one resource kind being accepted by a different endpoint.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM jobs WHERE share_token = ? AND is_public = 1 AND type = ?",
                (share_token, job_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM jobs WHERE share_token = ? AND is_public = 1",
                (share_token,),
            )
        row = cursor.fetchone()
        if row:
            return _parse_job_row(row)
        return None


def delete_job(job_uuid: str) -> bool:
    """Delete a job. Returns True if the job was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE uuid = ?", (job_uuid,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted job with UUID: {job_uuid}")
        return deleted


# ============ Agent Test Jobs Functions ============


def create_agent_test_job(
    agent_id: str,
    job_type: str,
    status: str = "in_progress",
    details: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new agent test job and return its UUID.

    Args:
        agent_id: UUID of the agent this job is for
        job_type: Type of job (llm-unit-test, llm-benchmark)
        status: Initial status (defaults to 'in_progress')
        details: JSON config needed to re-trigger the job if interrupted
        results: Initial results (usually None)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        job_uuid = str(uuid.uuid4())
        details_json = json.dumps(details) if details is not None else None
        results_json = json.dumps(results) if results is not None else None
        cursor.execute(
            """
            INSERT INTO agent_test_jobs (uuid, agent_id, type, status, details, results)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_uuid, agent_id, job_type, status, details_json, results_json),
        )
        conn.commit()
        logger.info(
            f"Created agent test job with UUID: {job_uuid}, type: {job_type}, agent: {agent_id}"
        )
        return job_uuid


def _parse_agent_test_job_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse an agent test job database row and deserialize JSON fields."""
    job = dict(row)
    if job.get("details"):
        job["details"] = json.loads(job["details"])
    if job.get("results"):
        job["results"] = json.loads(job["results"])
    return job


def get_agent_test_job(job_uuid: str) -> Optional[Dict[str, Any]]:
    """Get an agent test job by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_test_jobs WHERE uuid = ?", (job_uuid,))
        row = cursor.fetchone()
        if row:
            return _parse_agent_test_job_row(row)
        return None


def get_agent_test_jobs_for_agent(
    agent_id: str, job_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all agent test jobs for a specific agent, optionally filtered by type."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM agent_test_jobs WHERE agent_id = ? AND type = ? ORDER BY created_at DESC",
                (agent_id, job_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM agent_test_jobs WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            )
        rows = cursor.fetchall()
        return [_parse_agent_test_job_row(row) for row in rows]


def get_all_agent_test_jobs(job_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all agent test jobs, optionally filtered by type."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM agent_test_jobs WHERE type = ? ORDER BY created_at DESC",
                (job_type,),
            )
        else:
            cursor.execute("SELECT * FROM agent_test_jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [_parse_agent_test_job_row(row) for row in rows]


def get_agent_test_jobs_for_user(
    user_id: str, job_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all agent test jobs belonging to a user (across all their agents).

    Joins agent_test_jobs with agents so that each returned dict includes
    ``agent_name`` and ``agent_id`` alongside the normal job fields.
    Results are ordered newest-updated-first.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                """
                SELECT atj.*, a.name AS agent_name, a.uuid AS agent_id
                FROM agent_test_jobs atj
                JOIN agents a ON atj.agent_id = a.uuid
                WHERE a.user_id = ? AND a.deleted_at IS NULL AND atj.type = ?
                ORDER BY atj.updated_at DESC
                """,
                (user_id, job_type),
            )
        else:
            cursor.execute(
                """
                SELECT atj.*, a.name AS agent_name, a.uuid AS agent_id
                FROM agent_test_jobs atj
                JOIN agents a ON atj.agent_id = a.uuid
                WHERE a.user_id = ? AND a.deleted_at IS NULL
                ORDER BY atj.updated_at DESC
                """,
                (user_id,),
            )
        rows = cursor.fetchall()
        return [_parse_agent_test_job_row(row) for row in rows]


def get_pending_agent_test_jobs() -> List[Dict[str, Any]]:
    """Get all agent test jobs with status 'in_progress' (for recovery on restart)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_test_jobs WHERE status = 'in_progress' ORDER BY created_at ASC"
        )
        rows = cursor.fetchall()
        return [_parse_agent_test_job_row(row) for row in rows]


def get_queued_agent_test_jobs(
    job_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Get all agent test jobs with status 'queued', optionally filtered by job types.

    Returns jobs with user_id included (via agent ownership).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"""SELECT atj.*, a.user_id FROM agent_test_jobs atj
                    JOIN agents a ON atj.agent_id = a.uuid
                    WHERE atj.status = 'queued' AND atj.type IN ({placeholders})
                    ORDER BY atj.created_at ASC""",
                job_types,
            )
        else:
            cursor.execute(
                """SELECT atj.*, a.user_id FROM agent_test_jobs atj
                   JOIN agents a ON atj.agent_id = a.uuid
                   WHERE atj.status = 'queued'
                   ORDER BY atj.created_at ASC"""
            )
        rows = cursor.fetchall()
        return [_parse_agent_test_job_row(row) for row in rows]


def count_running_agent_test_jobs(job_types: Optional[List[str]] = None) -> int:
    """Count agent test jobs with status 'in_progress', optionally filtered by job types."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"SELECT COUNT(*) FROM agent_test_jobs WHERE status = 'in_progress' AND type IN ({placeholders})",
                job_types,
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM agent_test_jobs WHERE status = 'in_progress'"
            )
        return cursor.fetchone()[0]


def count_running_agent_test_jobs_for_user(
    user_id: str, job_types: Optional[List[str]] = None
) -> int:
    """Count agent test jobs with status 'in_progress' for a specific user (via agent ownership)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"""SELECT COUNT(*) FROM agent_test_jobs atj
                    JOIN agents a ON atj.agent_id = a.uuid
                    WHERE atj.status = 'in_progress' AND a.user_id = ? AND atj.type IN ({placeholders})""",
                [user_id] + job_types,
            )
        else:
            cursor.execute(
                """SELECT COUNT(*) FROM agent_test_jobs atj
                   JOIN agents a ON atj.agent_id = a.uuid
                   WHERE atj.status = 'in_progress' AND a.user_id = ?""",
                (user_id,),
            )
        return cursor.fetchone()[0]


def update_agent_test_job(
    job_uuid: str,
    status: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update an agent test job. Returns True if the job was found and updated."""
    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if results is not None:
        updates.append("results = ?")
        params.append(json.dumps(results))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(job_uuid)

    query = f"UPDATE agent_test_jobs SET {', '.join(updates)} WHERE uuid = ?"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated agent test job with UUID: {job_uuid}")
        return updated


def update_agent_test_job_visibility(
    job_uuid: str, is_public: bool, share_token: Optional[str]
) -> bool:
    """Update is_public and share_token for an agent test job. Returns True if found."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE agent_test_jobs SET is_public = ?, share_token = ?, updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
            (1 if is_public else 0, share_token, job_uuid),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_agent_test_job_by_share_token(
    share_token: str, job_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get an agent test job by its share_token, optionally restricted to a specific type.

    Always filters to is_public = 1. Pass job_type (e.g. 'llm-unit-test',
    'llm-benchmark') to prevent test-run tokens from resolving on the benchmark
    endpoint and vice versa.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM agent_test_jobs WHERE share_token = ? AND is_public = 1 AND type = ?",
                (share_token, job_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM agent_test_jobs WHERE share_token = ? AND is_public = 1",
                (share_token,),
            )
        row = cursor.fetchone()
        if row:
            return _parse_agent_test_job_row(row)
        return None


def delete_agent_test_job(job_uuid: str) -> bool:
    """Delete an agent test job. Returns True if the job was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM agent_test_jobs WHERE uuid = ?", (job_uuid,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted agent test job with UUID: {job_uuid}")
        return deleted


# ============ Simulation Jobs Functions ============


def create_simulation_job(
    simulation_id: str,
    job_type: str,
    status: str = "in_progress",
    details: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new simulation job and return its UUID.

    Args:
        simulation_id: UUID of the simulation this job is for
        job_type: Type of job (llm-simulation)
        status: Initial status (defaults to 'in_progress')
        details: JSON config needed to re-trigger the job if interrupted
        results: Initial results (usually None)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        job_uuid = str(uuid.uuid4())
        details_json = json.dumps(details) if details is not None else None
        results_json = json.dumps(results) if results is not None else None
        cursor.execute(
            """
            INSERT INTO simulation_jobs (uuid, simulation_id, type, status, details, results)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_uuid, simulation_id, job_type, status, details_json, results_json),
        )
        conn.commit()
        logger.info(
            f"Created simulation job with UUID: {job_uuid}, type: {job_type}, simulation: {simulation_id}"
        )
        return job_uuid


def _parse_simulation_job_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Parse a simulation job database row and deserialize JSON fields."""
    job = dict(row)
    if job.get("details"):
        job["details"] = json.loads(job["details"])
    if job.get("results"):
        job["results"] = json.loads(job["results"])
    return job


def get_simulation_job(job_uuid: str) -> Optional[Dict[str, Any]]:
    """Get a simulation job by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM simulation_jobs WHERE uuid = ?", (job_uuid,))
        row = cursor.fetchone()
        if row:
            return _parse_simulation_job_row(row)
        return None


def get_simulation_jobs_for_simulation(
    simulation_id: str, job_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all simulation jobs for a specific simulation, optionally filtered by type."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM simulation_jobs WHERE simulation_id = ? AND type = ? ORDER BY created_at DESC",
                (simulation_id, job_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM simulation_jobs WHERE simulation_id = ? ORDER BY created_at DESC",
                (simulation_id,),
            )
        rows = cursor.fetchall()
        return [_parse_simulation_job_row(row) for row in rows]


def get_all_simulation_jobs(job_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all simulation jobs, optionally filtered by type."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_type:
            cursor.execute(
                "SELECT * FROM simulation_jobs WHERE type = ? ORDER BY created_at DESC",
                (job_type,),
            )
        else:
            cursor.execute("SELECT * FROM simulation_jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [_parse_simulation_job_row(row) for row in rows]


def get_pending_simulation_jobs() -> List[Dict[str, Any]]:
    """Get all simulation jobs with status 'in_progress' (for recovery on restart)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_jobs WHERE status = 'in_progress' ORDER BY created_at ASC"
        )
        rows = cursor.fetchall()
        return [_parse_simulation_job_row(row) for row in rows]


def get_queued_simulation_jobs(
    job_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Get all simulation jobs with status 'queued', optionally filtered by job types.

    Returns jobs with user_id included (via simulation ownership).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"""SELECT sj.*, s.user_id FROM simulation_jobs sj
                    JOIN simulations s ON sj.simulation_id = s.uuid
                    WHERE sj.status = 'queued' AND sj.type IN ({placeholders})
                    ORDER BY sj.created_at ASC""",
                job_types,
            )
        else:
            cursor.execute(
                """SELECT sj.*, s.user_id FROM simulation_jobs sj
                   JOIN simulations s ON sj.simulation_id = s.uuid
                   WHERE sj.status = 'queued'
                   ORDER BY sj.created_at ASC"""
            )
        rows = cursor.fetchall()
        return [_parse_simulation_job_row(row) for row in rows]


def count_running_simulation_jobs(job_types: Optional[List[str]] = None) -> int:
    """Count simulation jobs with status 'in_progress', optionally filtered by job types."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"SELECT COUNT(*) FROM simulation_jobs WHERE status = 'in_progress' AND type IN ({placeholders})",
                job_types,
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM simulation_jobs WHERE status = 'in_progress'"
            )
        return cursor.fetchone()[0]


def count_running_simulation_jobs_for_user(
    user_id: str, job_types: Optional[List[str]] = None
) -> int:
    """Count simulation jobs with status 'in_progress' for a specific user (via simulation ownership)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            cursor.execute(
                f"""SELECT COUNT(*) FROM simulation_jobs sj
                    JOIN simulations s ON sj.simulation_id = s.uuid
                    WHERE sj.status = 'in_progress' AND s.user_id = ? AND sj.type IN ({placeholders})""",
                [user_id] + job_types,
            )
        else:
            cursor.execute(
                """SELECT COUNT(*) FROM simulation_jobs sj
                   JOIN simulations s ON sj.simulation_id = s.uuid
                   WHERE sj.status = 'in_progress' AND s.user_id = ?""",
                (user_id,),
            )
        return cursor.fetchone()[0]


def update_simulation_job(
    job_uuid: str,
    status: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a simulation job. Returns True if the job was found and updated.

    If details is provided, it will be merged with existing details (not replaced).
    """
    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if results is not None:
        updates.append("results = ?")
        params.append(json.dumps(results))

    # For details, we need to merge with existing details
    if details is not None:
        # First, fetch existing details
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT details FROM simulation_jobs WHERE uuid = ?", (job_uuid,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                existing_details = json.loads(row[0])
                # Merge new details into existing
                existing_details.update(details)
                details = existing_details
        updates.append("details = ?")
        params.append(json.dumps(details))

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(job_uuid)

    query = f"UPDATE simulation_jobs SET {', '.join(updates)} WHERE uuid = ?"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Updated simulation job with UUID: {job_uuid}")
        return updated


def update_simulation_job_visibility(
    job_uuid: str, is_public: bool, share_token: Optional[str]
) -> bool:
    """Update is_public and share_token for a simulation job. Returns True if found."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE simulation_jobs SET is_public = ?, share_token = ?, updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
            (1 if is_public else 0, share_token, job_uuid),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_simulation_job_by_share_token(share_token: str) -> Optional[Dict[str, Any]]:
    """Get a simulation job by its share_token."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM simulation_jobs WHERE share_token = ? AND is_public = 1",
            (share_token,),
        )
        row = cursor.fetchone()
        if row:
            return _parse_simulation_job_row(row)
        return None


def delete_simulation_job(job_uuid: str) -> bool:
    """Delete a simulation job. Returns True if the job was found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM simulation_jobs WHERE uuid = ?", (job_uuid,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted simulation job with UUID: {job_uuid}")
        return deleted


# ============ Dataset Functions ============


def create_dataset(name: str, dataset_type: str, user_id: str) -> str:
    """Create a new dataset and return its UUID."""
    if dataset_type not in ("stt", "tts"):
        raise ValueError("Dataset type must be 'stt' or 'tts'")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        dataset_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO datasets (uuid, name, type, user_id)
            VALUES (?, ?, ?, ?)
            """,
            (dataset_uuid, name, dataset_type, user_id),
        )
        conn.commit()
        logger.info(f"Created dataset with UUID: {dataset_uuid}")
        return dataset_uuid


def get_dataset(dataset_uuid: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a dataset by UUID, scoped to the authenticated user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM datasets WHERE uuid = ? AND user_id = ? AND deleted_at IS NULL",
            (dataset_uuid, user_id),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_datasets(
    user_id: str, dataset_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all datasets for a user, optionally filtered by type."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if dataset_type:
            cursor.execute(
                "SELECT * FROM datasets WHERE user_id = ? AND type = ? AND deleted_at IS NULL ORDER BY created_at DESC",
                (user_id, dataset_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM datasets WHERE user_id = ? AND deleted_at IS NULL ORDER BY created_at DESC",
                (user_id,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_dataset_item_counts(dataset_uuids: List[str]) -> Dict[str, int]:
    """Return a {dataset_uuid: active_item_count} map in a single query."""
    if not dataset_uuids:
        return {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in dataset_uuids)
        cursor.execute(
            f"SELECT dataset_id, COUNT(*) FROM dataset_items WHERE dataset_id IN ({placeholders}) AND deleted_at IS NULL GROUP BY dataset_id",
            dataset_uuids,
        )
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        for uid in dataset_uuids:
            counts.setdefault(uid, 0)
        return counts


def get_dataset_eval_counts(dataset_uuids: List[str]) -> Dict[str, int]:
    """Return a {dataset_uuid: eval_job_count} map by reading the dataset_id stored in job details."""
    if not dataset_uuids:
        return {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in dataset_uuids)
        cursor.execute(
            f"SELECT json_extract(details, '$.dataset_id') AS ds_id, COUNT(*) FROM jobs"
            f" WHERE json_extract(details, '$.dataset_id') IN ({placeholders})"
            f" GROUP BY ds_id",
            dataset_uuids,
        )
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        for uid in dataset_uuids:
            counts.setdefault(uid, 0)
        return counts


def get_active_dataset_ids(dataset_uuids: List[str]) -> set:
    """Return the subset of dataset UUIDs that exist and are not soft-deleted."""
    if not dataset_uuids:
        return set()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in dataset_uuids)
        cursor.execute(
            f"SELECT uuid FROM datasets WHERE uuid IN ({placeholders}) AND deleted_at IS NULL",
            dataset_uuids,
        )
        return {row[0] for row in cursor.fetchall()}


def update_dataset_name(dataset_uuid: str, user_id: str, name: str) -> bool:
    """Rename a dataset. Returns True if found and updated."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE datasets SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE uuid = ? AND user_id = ? AND deleted_at IS NULL",
            (name, dataset_uuid, user_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Renamed dataset {dataset_uuid}")
        return updated


def delete_dataset(dataset_uuid: str, user_id: str) -> bool:
    """Soft delete a dataset and all its items. Returns True if found and deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE datasets SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE uuid = ? AND user_id = ? AND deleted_at IS NULL",
            (dataset_uuid, user_id),
        )
        if cursor.rowcount == 0:
            return False
        # Soft delete all items belonging to this dataset
        cursor.execute(
            "UPDATE dataset_items SET deleted_at = CURRENT_TIMESTAMP WHERE dataset_id = ? AND deleted_at IS NULL",
            (dataset_uuid,),
        )
        conn.commit()
        logger.info(f"Soft deleted dataset {dataset_uuid} and its items")
        return True


def add_dataset_items(
    dataset_id: str,
    items: List[Dict[str, Any]],
) -> List[str]:
    """Add items to a dataset. Returns list of new item UUIDs.

    Each item dict must have 'text' and optionally 'audio_path'.
    order_index is assigned sequentially after the current max, preserving
    existing order even across multiple bulk inserts.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Find the current max order_index for this dataset (including soft-deleted
        # rows so that restored items never collide with new ones)
        cursor.execute(
            "SELECT COALESCE(MAX(order_index), -1) FROM dataset_items WHERE dataset_id = ?",
            (dataset_id,),
        )
        max_index = cursor.fetchone()[0]

        item_uuids = []
        for offset, item in enumerate(items):
            item_uuid = str(uuid.uuid4())
            order_index = max_index + 1 + offset
            cursor.execute(
                """
                INSERT INTO dataset_items (uuid, dataset_id, audio_path, text, order_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item_uuid,
                    dataset_id,
                    item.get("audio_path"),
                    str(item["text"]),
                    order_index,
                ),
            )
            item_uuids.append(item_uuid)

        if item_uuids:
            cursor.execute(
                "UPDATE datasets SET updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                (dataset_id,),
            )
        conn.commit()
        logger.info(f"Added {len(item_uuids)} items to dataset {dataset_id}")
        return item_uuids


def get_dataset_item(item_uuid: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    """Get a single active dataset item by UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM dataset_items WHERE uuid = ? AND dataset_id = ? AND deleted_at IS NULL",
            (item_uuid, dataset_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        item = dict(row)
        item["text"] = str(item["text"])
        return item


def get_dataset_items(dataset_id: str) -> List[Dict[str, Any]]:
    """Get all active items for a dataset, ordered by order_index."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM dataset_items WHERE dataset_id = ? AND deleted_at IS NULL ORDER BY order_index ASC",
            (dataset_id,),
        )
        rows = cursor.fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            item["text"] = str(item["text"])
        return items


def get_dataset_items_by_uuids(item_uuids: List[str]) -> List[Dict[str, Any]]:
    """Fetch specific dataset items by UUID, ordered by order_index."""
    if not item_uuids:
        return []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in item_uuids)
        cursor.execute(
            f"SELECT * FROM dataset_items WHERE uuid IN ({placeholders}) AND deleted_at IS NULL ORDER BY order_index ASC",
            item_uuids,
        )
        items = [dict(row) for row in cursor.fetchall()]
        for item in items:
            item["text"] = str(item["text"])
        return items


def update_dataset_item(
    item_uuid: str,
    dataset_id: str,
    text: Optional[str] = None,
    audio_path: Optional[str] = ...,
) -> bool:
    """Update a dataset item's text and/or audio_path. Returns True if found and updated.

    audio_path uses sentinel default (...) so callers can explicitly pass None to clear it.
    """
    fields = []
    params: list = []
    if text is not None:
        fields.append("text = ?")
        params.append(text)
    if audio_path is not ...:
        fields.append("audio_path = ?")
        params.append(audio_path)
    if not fields:
        return False
    fields.append("updated_at = CURRENT_TIMESTAMP")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        params.extend([item_uuid, dataset_id])
        cursor.execute(
            f"UPDATE dataset_items SET {', '.join(fields)} WHERE uuid = ? AND dataset_id = ? AND deleted_at IS NULL",
            params,
        )
        updated = cursor.rowcount > 0
        if updated:
            cursor.execute(
                "UPDATE datasets SET updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                (dataset_id,),
            )
        conn.commit()
        return updated


def delete_dataset_item(item_uuid: str, dataset_id: str) -> bool:
    """Soft delete a single dataset item. Returns True if found and deleted.

    order_index values of remaining items are intentionally not renumbered —
    ORDER BY order_index on the filtered (deleted_at IS NULL) set still
    produces the correct relative order with gaps.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE dataset_items SET deleted_at = CURRENT_TIMESTAMP WHERE uuid = ? AND dataset_id = ? AND deleted_at IS NULL",
            (item_uuid, dataset_id),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            cursor.execute(
                "UPDATE datasets SET updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                (dataset_id,),
            )
            logger.info(f"Soft deleted dataset item {item_uuid}")
        conn.commit()
        return deleted


# ============ User Limits Functions ============


def create_user_limits(user_id: str, limits: "UserLimits") -> str:
    """Create a user limits row. Returns the UUID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        row_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO user_limits (uuid, user_id, limits)
            VALUES (?, ?, ?)
            """,
            (row_uuid, user_id, limits.model_dump_json()),
        )
        conn.commit()
        logger.info(f"Created user_limits for user {user_id} with UUID: {row_uuid}")
        return row_uuid


def get_user_limits(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user limits by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_limits WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["limits"] = json.loads(result["limits"])
            return result
        return None


def update_user_limits(user_id: str, limits: "UserLimits") -> Optional[Dict[str, Any]]:
    """Update limits JSON for a user. Returns the updated row, or None if not found."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE user_limits SET limits = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (limits.model_dump_json(), user_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
        cursor.execute(
            "SELECT * FROM user_limits WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["limits"] = json.loads(result["limits"])
            return result
        return None


def delete_user_limits(user_id: str) -> bool:
    """Delete user limits row. Returns True if deleted."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_limits WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
