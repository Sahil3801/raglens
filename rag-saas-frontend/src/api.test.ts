import { beforeEach, describe, expect, it, vi } from 'vitest';

describe('API client', () => {
  beforeEach(() => vi.resetModules());

  it.each(['', 'http://localhost:8001/'])('uses the configured base URL %s', async (base) => {
    vi.stubEnv('VITE_API_BASE_URL', base);
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ documents: ['a.pdf'] })));
    vi.stubGlobal('fetch', fetch);
    const { apiRequest } = await import('./api');
    await expect(apiRequest('/documents')).resolves.toEqual({ documents: ['a.pdf'] });
    expect(fetch).toHaveBeenCalledWith(`${base ? 'http://localhost:8001' : 'http://127.0.0.1:8000'}/documents`, undefined);
  });

  it('preserves multipart body and lets the browser set its boundary', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{}'));
    vi.stubGlobal('fetch', fetch);
    const form = new FormData();
    form.append('file', new File(['pdf'], 'a.pdf'));
    const { apiRequest } = await import('./api');
    await apiRequest('/upload', { method: 'POST', body: form });
    expect(fetch.mock.calls[0][1]).toEqual({ method: 'POST', body: form });
  });

  it.each([
    [400, JSON.stringify({ detail: 'The PDF cannot be read.' }), 'The PDF cannot be read.'],
    [422, JSON.stringify({ detail: [{ loc: ['body', 'query'] }] }), 'Check the backend logs.'],
    [502, '<html>Bad gateway</html>', 'Check the backend logs.'],
  ])('rejects HTTP %s rather than returning error data as success', async (status, body, detail) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, { status })));
    const { apiRequest } = await import('./api');
    await expect(apiRequest('/chat')).rejects.toThrow(`Request failed (HTTP ${status}). ${detail}`);
  });

  it('preserves network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    const { apiRequest } = await import('./api');
    await expect(apiRequest('/documents')).rejects.toThrow('Failed to fetch');
  });

  it('rejects invalid JSON in successful responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('not JSON')));
    const { apiRequest } = await import('./api');
    await expect(apiRequest('/documents')).rejects.toThrow();
  });

  it('renders Error messages and a safe fallback for non-errors', async () => {
    const { errorMessage } = await import('./api');
    expect(errorMessage(new Error('offline'))).toBe('offline');
    expect(errorMessage({ secret: 'must not be displayed' })).toBe('The request failed.');
  });
});
