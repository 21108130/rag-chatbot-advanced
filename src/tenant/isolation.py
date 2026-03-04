
from __future__ import annotations

import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import logger


class IsolationLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


ISOLATION_DESCRIPTIONS = {
    IsolationLevel.HIGH:   "Separate database per tenant — full data isolation",
    IsolationLevel.MEDIUM: "Separate collection per tenant — logical isolation",
    IsolationLevel.LOW:    "Shared collection with tenant_id filter — cost-efficient",
}

ISOLATION_COMPLIANCE = {
    IsolationLevel.HIGH:   ["HIPAA", "SOC2", "FedRAMP", "GDPR"],
    IsolationLevel.MEDIUM: ["GDPR", "SOC2"],
    IsolationLevel.LOW:    ["Basic"],
}


class TenantInfo:


    def __init__(
        self,
        tenant_id:       str,
        isolation_level: IsolationLevel,
        created_at:      datetime,
        plan:            str = "standard",
        metadata:        Optional[Dict] = None,
    ) -> None:
        self.tenant_id       = tenant_id
        self.isolation_level = isolation_level
        self.created_at      = created_at
        self.plan            = plan
        self.metadata        = metadata or {}
        self.doc_count:  int = 0
        self.query_count: int = 0




class TenantManager:


    def __init__(
        self,
        default_isolation: IsolationLevel = IsolationLevel.MEDIUM,
        base_chroma_dir:   str             = "./data/chroma_db",
    ) -> None:
        self.default_isolation = default_isolation
        self.base_chroma_dir   = Path(base_chroma_dir)
        self._tenants: Dict[str, TenantInfo] = {}
        self._stores:  Dict[str, Any]        = {}
        logger.info(
            f"[TenantManager] Initialized (default_isolation={default_isolation})"
        )

    def register_tenant(
        self,
        tenant_id:       str,
        isolation_level: Optional[IsolationLevel] = None,
        plan:            str                       = "standard",
        metadata:        Optional[Dict]            = None,
    ) -> TenantInfo:

        if tenant_id in self._tenants:
            logger.info(f"[TenantManager] Tenant '{tenant_id}' already registered")
            return self._tenants[tenant_id]

        level = isolation_level or self.default_isolation


        if plan == "enterprise" and level == IsolationLevel.LOW:
            level = IsolationLevel.MEDIUM
            logger.info(
                f"[TenantManager] Upgrading '{tenant_id}' to MEDIUM isolation "
                f"(enterprise plan)"
            )

        info = TenantInfo(
            tenant_id       = tenant_id,
            isolation_level = level,
            created_at      = datetime.utcnow(),
            plan            = plan,
            metadata        = metadata or {},
        )
        self._tenants[tenant_id] = info

        logger.info(
            f"[TenantManager] Registered tenant '{tenant_id}' "
            f"(level={level}, plan={plan})"
        )
        return info

    def get_vector_store(self, tenant_id: str):

        if tenant_id not in self._tenants:

            self.register_tenant(tenant_id)

        if tenant_id in self._stores:
            return self._stores[tenant_id]

        info  = self._tenants[tenant_id]
        store = self._create_store(tenant_id, info.isolation_level)
        self._stores[tenant_id] = store

        logger.info(
            f"[TenantManager] Created store for '{tenant_id}' "
            f"(level={info.isolation_level})"
        )
        return store

    def _create_store(self, tenant_id: str, level: IsolationLevel):
        """Create a ChromaVectorStore with the appropriate isolation."""
        from src.vectordb.chroma_store import ChromaVectorStore

        safe_id = self._safe_id(tenant_id)

        if level == IsolationLevel.HIGH:

            tenant_dir = self.base_chroma_dir / "tenants" / safe_id
            tenant_dir.mkdir(parents=True, exist_ok=True)
            store = ChromaVectorStore(
                collection_name = f"docs_{safe_id}",
                persist_directory = str(tenant_dir),
            )
            logger.info(f"[TenantManager] HIGH isolation: {tenant_dir}")

        elif level == IsolationLevel.MEDIUM:

            store = ChromaVectorStore(
                collection_name = f"tenant_{safe_id}",
            )
            logger.info(f"[TenantManager] MEDIUM isolation: collection=tenant_{safe_id}")

        else:
            store = ChromaVectorStore(
                collection_name = "shared_tenant_collection",
            )
            logger.info(f"[TenantManager] LOW isolation: shared collection + filter")

        return store

    def get_where_filter(self, tenant_id: str) -> Optional[Dict]:

        if tenant_id not in self._tenants:
            return None
        info = self._tenants[tenant_id]
        if info.isolation_level == IsolationLevel.LOW:
            return {"tenant_id": tenant_id}
        return None

    def delete_tenant(self, tenant_id: str) -> bool:

        if tenant_id not in self._tenants:
            logger.warning(f"[TenantManager] Cannot delete unknown tenant '{tenant_id}'")
            return False

        info  = self._tenants[tenant_id]
        safe  = self._safe_id(tenant_id)

        try:
            if info.isolation_level == IsolationLevel.HIGH:
                tenant_dir = self.base_chroma_dir / "tenants" / safe
                if tenant_dir.exists():
                    shutil.rmtree(tenant_dir)
                    logger.info(f"[TenantManager] Deleted HIGH isolation dir: {tenant_dir}")

            elif info.isolation_level == IsolationLevel.MEDIUM:
                if tenant_id in self._stores:
                    try:
                        store = self._stores[tenant_id]
                        store.delete_collection()
                    except Exception as e:
                        logger.warning(f"[TenantManager] Collection delete failed: {e}")


            self._tenants.pop(tenant_id, None)
            self._stores.pop(tenant_id, None)

            logger.info(f"[TenantManager] Deleted tenant '{tenant_id}'")
            return True

        except Exception as e:
            logger.error(f"[TenantManager] Delete failed for '{tenant_id}': {e}")
            return False

    def list_tenants(self) -> List[Dict]:
        """Return summary of all registered tenants."""
        return [
            {
                "tenant_id":       t.tenant_id,
                "isolation_level": t.isolation_level.value,
                "plan":            t.plan,
                "created_at":      str(t.created_at),
                "doc_count":       t.doc_count,
                "query_count":     t.query_count,
                "compliance":      ISOLATION_COMPLIANCE[t.isolation_level],
            }
            for t in self._tenants.values()
        ]

    def get_tenant_info(self, tenant_id: str) -> Optional[Dict]:
        """Get info for a specific tenant."""
        if tenant_id not in self._tenants:
            return None
        t = self._tenants[tenant_id]
        return {
            "tenant_id":       t.tenant_id,
            "isolation_level": t.isolation_level.value,
            "isolation_description": ISOLATION_DESCRIPTIONS[t.isolation_level],
            "plan":            t.plan,
            "created_at":      str(t.created_at),
            "doc_count":       t.doc_count,
            "query_count":     t.query_count,
            "compliance":      ISOLATION_COMPLIANCE[t.isolation_level],
            "metadata":        t.metadata,
        }

    def increment_doc_count(self, tenant_id: str, by: int = 1) -> None:
        if tenant_id in self._tenants:
            self._tenants[tenant_id].doc_count += by

    def increment_query_count(self, tenant_id: str) -> None:
        if tenant_id in self._tenants:
            self._tenants[tenant_id].query_count += 1

    @staticmethod
    def _safe_id(tenant_id: str) -> str:
        """Create a filesystem-safe tenant identifier."""
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '_', tenant_id)[:32]


_manager: Optional[TenantManager] = None

def get_tenant_manager(
    isolation_level: IsolationLevel = IsolationLevel.MEDIUM,
) -> TenantManager:
    global _manager
    if _manager is None:
        _manager = TenantManager(default_isolation=isolation_level)
    return _manager
