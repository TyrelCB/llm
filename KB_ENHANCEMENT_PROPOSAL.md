# KB Article Update Handling - Implementation Plan

Based on your preferences for **Smart Update** behavior, **Keep Latest Only** versioning, and **Smart Detection**, here's the comprehensive implementation plan:

## 1. Enhanced Duplicate Detection & Update Logic

### Core Changes to `src/knowledge/vectorstore.py`:

**New Methods:**
```python
def _find_by_source_or_hash(self, document: Document) -> dict | None:
    """Find existing document by source path or content hash."""
    
def _should_update_document(self, existing: dict, new_doc: Document, update_mode: str) -> tuple[bool, str]:
    """Determine if document should be updated based on mode and changes."""
    
def _update_document(self, old_doc_id: str, new_doc: Document) -> str:
    """Replace existing document with new version."""
```

**Smart Detection Logic:**
- Check file modification time (mtime) vs stored `last_updated`
- Compare file size for quick change detection
- Use content hash for definitive comparison
- Track file metadata for change verification

### Update Modes Configuration:
**Add to `config/settings.py`:**
```python
kb_update_mode: Literal["smart", "force", "skip"] = "smart"
kb_track_file_metadata: bool = True
```

## 2. Enhanced Metadata Schema

**New Metadata Fields:**
```python
{
    # Existing fields
    "content_hash": "...",
    "source": "...",
    "added_at": "...",
    
    # New fields for update handling
    "last_updated": "...",        # Last modification timestamp
    "file_mtime": 1706091234.5,   # File modification time
    "file_size": 1024,            # File size in bytes
    "update_version": 2,          # Increment on updates
    "source_hash": "...",         # Hash of source path for tracking moves
}
```

## 3. CLI Interface Enhancements

**New CLI Options in `scripts/ingest.py`:**

```bash
# Update-aware ingestion
python scripts/ingest.py file document.md --update        # Smart update
python scripts/ingest.py file document.md --force-update   # Force replace
python scripts/ingest.py directory ./docs --smart-update   # Smart mode
python scripts/ingest.py directory ./docs --check-updates  # Dry run

# Update-specific commands
python scripts/ingest.py check-updates ./docs              # Show what would change
python scripts/ingest.py update-source ./docs              # Update specific source
```

**Enhanced Statistics:**
- Show files updated vs skipped
- Display version counts
- Report on changes detected

## 4. Ingestion Pipeline Changes

### Modified `IngestionPipeline` Methods:

**`ingest_file()` Enhancement:**
- Add `update_mode` parameter
- Pass file metadata (mtime, size) to loader
- Report update statistics

**`ingest_directory()` Enhancement:**
- Collect file metadata before processing
- Option to only process changed files
- Efficient batch updates

### Enhanced `DocumentLoader`:
**File metadata collection in `_load_text()` and `_load_pdf()`:**
```python
metadata.update({
    "file_mtime": file_path.stat().st_mtime,
    "file_size": file_path.stat().st_size,
    "source_hash": hashlib.md5(str(file_path).encode()).hexdigest()[:8],
})
```

## 5. Implementation Roadmap

### Phase 1: Core Update Logic (Priority: High)
1. **VectorStore Updates:**
   - Implement `_find_by_source_or_hash()`
   - Add `_should_update_document()` with smart detection
   - Create `_update_document()` method
   - Modify `add_documents()` for update-aware processing

2. **Configuration:**
   - Add update settings to `config/settings.py`
   - Implement update mode validation

### Phase 2: Pipeline Integration (Priority: High)
1. **IngestionPipeline:**
   - Add update mode parameters
   - Integrate change detection
   - Enhance statistics reporting

2. **DocumentLoader:**
   - Collect file metadata
   - Pass metadata through pipeline

### Phase 3: CLI Enhancements (Priority: Medium)
1. **New CLI Options:**
   - Add update flags to existing commands
   - Implement check-updates command
   - Add update-source command

2. **Enhanced Reporting:**
   - Update statistics display
   - Add verbose logging for updates

### Phase 4: Advanced Features (Priority: Low)
1. **Batch Operations:**
   - Efficient directory-wide updates
   - Parallel processing for large updates

2. **Monitoring:**
   - Update history tracking
   - Performance metrics

## 6. Key Implementation Details

### Smart Detection Algorithm:
```python
def _document_changed(self, existing: dict, new_doc: Document) -> bool:
    """Comprehensive change detection."""
    new_metadata = new_doc.metadata
    
    # 1. Content hash comparison (definitive)
    if existing["content_hash"] != new_metadata.get("content_hash"):
        return True
    
    # 2. File metadata changes (quick check)
    if (existing.get("file_mtime") != new_metadata.get("file_mtime") or
        existing.get("file_size") != new_metadata.get("file_size")):
        return True
    
    return False
```

### Update Decision Matrix:
| Scenario | Smart Mode | Force Mode | Skip Mode |
|----------|------------|------------|-----------|
| Same source, same content | Skip | Update | Skip |
| Same source, changed content | Update | Update | Skip |
| Different source, same content | Skip | Update | Skip |
| Different source, different content | Add | Update | Add |

## 7. Backward Compatibility

- Existing KBs work unchanged (new metadata fields optional)
- Default behavior maintains current functionality
- Gradual migration path for existing documents

## 8. Testing Strategy

- Unit tests for update detection logic
- Integration tests for CLI commands
- Performance tests for large directory updates
- Backward compatibility verification

## 9. Configuration Examples

### Environment Variables
```bash
# .env file
KB_UPDATE_MODE=smart
KB_TRACK_FILE_METADATA=true
```

### Programmatic Configuration
```python
from config.settings import settings

# Override update mode for specific operation
settings.kb_update_mode = "force"

# Disable file metadata tracking for faster ingestion
settings.kb_track_file_metadata = False
```

## 10. Migration Strategy

### Phase 1: Non-breaking changes
- Add new metadata fields to newly ingested documents
- Maintain existing duplicate detection as fallback

### Phase 2: Gradual migration
- Provide migration script to add metadata to existing documents
- Allow mixed-mode operation during transition

### Phase 3: Full operation
- All update features available
- Optional cleanup of legacy documents

## 11. Performance Considerations

### Optimizations:
- Batch metadata collection for directories
- Early exit for unchanged files (mtime check)
- Efficient ChromaDB queries with proper indexing
- Parallel processing for large updates

### Benchmarks to Target:
- < 100ms per document for update detection
- < 10s for 1000-document directory update
- < 1s for single document update verification

## 12. Error Handling & Edge Cases

### Scenarios to Handle:
- File deletion between detection and ingestion
- Permission errors during metadata collection
- Corrupted ChromaDB entries
- Concurrent ingestion conflicts

### Recovery Strategies:
- Graceful degradation to current behavior
- Transactional updates where possible
- Comprehensive logging for debugging

This plan provides a robust, configurable update system while maintaining the simplicity and efficiency of the current design.