"""
MaxQ Component Generator

Generates embeddable React components for search functionality.
"""

REACT_SEARCH_TEMPLATE = '''// MaxQ Search Component
// Generated for project: {project_name}
// Project ID: {project_id}
// API URL: {api_url}
//
// Usage:
//   import {{ MaxQSearch }} from './MaxQSearch';
//   <MaxQSearch />

import React, {{ useState, useCallback }} from 'react';

export function MaxQSearch({{
  apiUrl = "{api_url}",
  projectId = "{project_id}",
  placeholder = "Search...",
  limit = 10,
  strategy = "hybrid",
  className = "",
  onResultClick
}}) {{
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const search = useCallback(async () => {{
    if (!query.trim()) {{
      setResults([]);
      return;
    }}

    setLoading(true);
    setError(null);

    try {{
      const response = await fetch(`${{apiUrl}}/playground/search`, {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
          // Add API key header if needed:
          // 'X-API-Key': 'your-api-key'
        }},
        body: JSON.stringify({{
          project_id: projectId,
          query: query,
          strategy: strategy,
          limit: limit
        }})
      }});

      if (!response.ok) {{
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Search failed');
      }}

      const data = await response.json();
      setResults(data.results || []);
    }} catch (err) {{
      console.error('MaxQ Search error:', err);
      setError(err.message);
      setResults([]);
    }} finally {{
      setLoading(false);
    }}
  }}, [query, apiUrl, projectId, strategy, limit]);

  const handleKeyDown = (e) => {{
    if (e.key === 'Enter') {{
      search();
    }}
  }};

  const handleResultClick = (result, index) => {{
    if (onResultClick) {{
      onResultClick(result, index);
    }}
  }};

  return (
    <div className={{`maxq-search ${{className}}`}} style={{{{ fontFamily: 'system-ui, sans-serif' }}}}>
      <div className="maxq-search-input-wrapper" style={{{{
        display: 'flex',
        gap: '8px',
        marginBottom: '16px'
      }}}}>
        <input
          type="text"
          value={{query}}
          onChange={{(e) => setQuery(e.target.value)}}
          onKeyDown={{handleKeyDown}}
          placeholder={{placeholder}}
          disabled={{loading}}
          style={{{{
            flex: 1,
            padding: '12px 16px',
            fontSize: '16px',
            border: '1px solid #e0e0e0',
            borderRadius: '8px',
            outline: 'none',
            transition: 'border-color 0.2s',
          }}}}
        />
        <button
          onClick={{search}}
          disabled={{loading || !query.trim()}}
          style={{{{
            padding: '12px 24px',
            fontSize: '16px',
            fontWeight: '500',
            color: '#fff',
            backgroundColor: loading ? '#999' : '#2563eb',
            border: 'none',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s',
          }}}}
        >
          {{loading ? 'Searching...' : 'Search'}}
        </button>
      </div>

      {{error && (
        <div className="maxq-search-error" style={{{{
          padding: '12px',
          marginBottom: '16px',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '8px',
          color: '#dc2626',
          fontSize: '14px'
        }}}}>
          {{error}}
        </div>
      )}}

      {{results.length > 0 && (
        <ul className="maxq-search-results" style={{{{
          listStyle: 'none',
          padding: 0,
          margin: 0
        }}}}>
          {{results.map((result, index) => (
            <li
              key={{result.id || index}}
              className="maxq-search-result"
              onClick={{() => handleResultClick(result, index)}}
              style={{{{
                padding: '16px',
                borderBottom: '1px solid #f0f0f0',
                cursor: onResultClick ? 'pointer' : 'default',
                transition: 'background-color 0.15s',
              }}}}
              onMouseEnter={{(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}}
              onMouseLeave={{(e) => e.currentTarget.style.backgroundColor = 'transparent'}}
            >
              <div style={{{{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}}}>
                <span className="maxq-result-text" style={{{{
                  flex: 1,
                  fontSize: '14px',
                  lineHeight: '1.5',
                  color: '#374151'
                }}}}>
                  {{result.text}}
                </span>
                <span className="maxq-result-score" style={{{{
                  marginLeft: '16px',
                  fontSize: '12px',
                  fontFamily: 'monospace',
                  color: '#6b7280',
                  backgroundColor: '#f3f4f6',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}}}>
                  {{result.score?.toFixed(3)}}
                </span>
              </div>
              {{result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="maxq-result-metadata" style={{{{
                  marginTop: '8px',
                  fontSize: '12px',
                  color: '#9ca3af'
                }}}}>
                  {{Object.entries(result.metadata).slice(0, 3).map(([key, value]) => (
                    <span key={{key}} style={{{{ marginRight: '12px' }}}}>
                      {{key}}: {{String(value).substring(0, 50)}}
                    </span>
                  ))}}
                </div>
              )}}
            </li>
          ))}}
        </ul>
      )}}

      {{!loading && results.length === 0 && query && (
        <div className="maxq-search-empty" style={{{{
          textAlign: 'center',
          padding: '32px',
          color: '#9ca3af',
          fontSize: '14px'
        }}}}>
          No results found for "{{query}}"
        </div>
      )}}
    </div>
  );
}}

// Optional: CSS-in-JS styles that can be added to a stylesheet
export const styles = `
.maxq-search {{
  max-width: 600px;
}}

.maxq-search input:focus {{
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}}

.maxq-search button:hover:not(:disabled) {{
  background-color: #1d4ed8;
}}

.maxq-search-result {{
  animation: fadeIn 0.2s ease-out;
}}

@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(-4px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
`;

export default MaxQSearch;
'''


def generate_react_component(
    project_name: str,
    project_id: str,
    api_url: str = "http://localhost:8000"
) -> str:
    """
    Generate a React search component for a MaxQ project.

    Args:
        project_name: Human-readable project name
        project_id: MaxQ project ID
        api_url: URL of the MaxQ API server

    Returns:
        React component code as string
    """
    return REACT_SEARCH_TEMPLATE.format(
        project_name=project_name,
        project_id=project_id,
        api_url=api_url
    )


# TypeScript version for modern React projects
TYPESCRIPT_SEARCH_TEMPLATE = '''// MaxQ Search Component (TypeScript)
// Generated for project: {project_name}
// Project ID: {project_id}

import React, {{ useState, useCallback }} from 'react';

interface SearchResult {{
  id: string;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}}

interface MaxQSearchProps {{
  apiUrl?: string;
  projectId?: string;
  placeholder?: string;
  limit?: number;
  strategy?: 'hybrid' | 'dense' | 'sparse';
  className?: string;
  onResultClick?: (result: SearchResult, index: number) => void;
}}

export const MaxQSearch: React.FC<MaxQSearchProps> = ({{
  apiUrl = "{api_url}",
  projectId = "{project_id}",
  placeholder = "Search...",
  limit = 10,
  strategy = "hybrid",
  className = "",
  onResultClick
}}) => {{
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(async () => {{
    if (!query.trim()) {{
      setResults([]);
      return;
    }}

    setLoading(true);
    setError(null);

    try {{
      const response = await fetch(`${{apiUrl}}/playground/search`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          project_id: projectId,
          query,
          strategy,
          limit
        }})
      }});

      if (!response.ok) {{
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Search failed');
      }}

      const data = await response.json();
      setResults(data.results || []);
    }} catch (err) {{
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
    }} finally {{
      setLoading(false);
    }}
  }}, [query, apiUrl, projectId, strategy, limit]);

  return (
    <div className={{`maxq-search ${{className}}`}}>
      {{/* Component JSX same as JavaScript version */}}
    </div>
  );
}};

export default MaxQSearch;
'''


def generate_typescript_component(
    project_name: str,
    project_id: str,
    api_url: str = "http://localhost:8000"
) -> str:
    """Generate TypeScript version of the search component."""
    return TYPESCRIPT_SEARCH_TEMPLATE.format(
        project_name=project_name,
        project_id=project_id,
        api_url=api_url
    )
