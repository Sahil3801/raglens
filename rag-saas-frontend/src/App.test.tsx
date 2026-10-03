import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import App from './App';

const fetchMock = vi.fn<typeof fetch>();
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

function uploadInput() {
  return screen.getByLabelText('+ Upload PDF') as HTMLInputElement;
}

async function ready(names: string[] = []) {
  fetchMock.mockResolvedValueOnce(json({ documents: names }));
  const user = userEvent.setup();
  render(<App />);
  if (names.length) await screen.findByText(names[0]);
  else await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  return user;
}

it('loads and displays documents in global mode', async () => {
  await ready(['a.pdf', 'b.pdf']);
  expect(screen.getByText('b.pdf')).toBeInTheDocument();
  expect(screen.getByText('Global Search (All Documents)')).toBeInTheDocument();
});

it('shows initial load failures instead of treating them as success', async () => {
  fetchMock.mockRejectedValueOnce(new Error('Backend offline'));
  render(<App />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Backend offline');
});

it('sends global and selected-document payloads and toggles selection off', async () => {
  const user = await ready(['a.pdf']);
  await user.type(screen.getByRole('textbox'), 'Which code?');
  fetchMock.mockResolvedValueOnce(json({ answer: 'Global answer', sources: [] }));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  await screen.findByText('Global answer');
  expect(JSON.parse(fetchMock.mock.calls[1][1]!.body as string)).toEqual({ query: 'Which code?' });
  await user.click(screen.getByText('a.pdf'));
  fetchMock.mockResolvedValueOnce(json({ answer: 'Filtered answer', sources: ['evidence'] }));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  await screen.findByText('Filtered answer');
  expect(JSON.parse(fetchMock.mock.calls[2][1]!.body as string)).toEqual({ query: 'Which code?', filter_filename: 'a.pdf' });
  await user.click(screen.getByText('a.pdf'));
  expect(screen.getByText('Global Search (All Documents)')).toBeInTheDocument();
});

it.each(['', '   '])('does not submit an empty/whitespace question %j', async (question) => {
  const user = await ready();
  fireEvent.change(screen.getByRole('textbox'), { target: { value: question } });
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it('shows chat failure and clears Thinking rather than leaving stale success', async () => {
  const user = await ready();
  await user.type(screen.getByRole('textbox'), 'Question');
  fetchMock.mockResolvedValueOnce(json({ detail: 'Provider unavailable' }, 503));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Provider unavailable');
  expect(screen.queryByText('Thinking...')).not.toBeInTheDocument();
  expect(screen.queryByText('Answer:')).not.toBeInTheDocument();
});

it('prevents duplicate requests while an answer is pending', async () => {
  const user = await ready();
  await user.type(screen.getByRole('textbox'), 'Question');
  let resolve!: (value: Response) => void;
  fetchMock.mockImplementationOnce(() => new Promise<Response>((done) => { resolve = done; }));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  expect(screen.getByText('Thinking...')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled();
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  expect(fetchMock).toHaveBeenCalledTimes(2);
  await act(async () => resolve(json({ answer: 'Completed', sources: [] })));
  expect(await screen.findByText('Completed')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Ask' })).toBeEnabled();
});

it('uploads multipart PDF, refreshes the list and selects the new document', async () => {
  const user = await ready();
  const file = new File(['%PDF test'], 'new.pdf', { type: 'application/pdf' });
  fetchMock.mockResolvedValueOnce(json({ filename: file.name, message: 'indexed' }))
    .mockResolvedValueOnce(json({ documents: [file.name] }));
  await user.upload(uploadInput(), file);
  expect(await screen.findByText('Filtering by new.pdf')).toBeInTheDocument();
  const request = fetchMock.mock.calls[1][1]!;
  expect(request.method).toBe('POST');
  expect((request.body as FormData).get('file')).toBe(file);
  expect(uploadInput()).toBeEnabled();
});

it('allows retrying the same file after an upload failure', async () => {
  const user = await ready();
  const file = new File(['%PDF test'], 'retry.pdf', { type: 'application/pdf' });
  fetchMock.mockResolvedValueOnce(json({ detail: 'Temporary failure' }, 503));
  await user.upload(uploadInput(), file);
  expect(await screen.findByRole('alert')).toHaveTextContent('Temporary failure');
  expect(screen.queryByText('Filtering by retry.pdf')).not.toBeInTheDocument();
  fetchMock.mockResolvedValueOnce(json({ filename: file.name, message: 'indexed' }))
    .mockResolvedValueOnce(json({ documents: [file.name] }));
  await user.upload(uploadInput(), file);
  expect(await screen.findByText('Filtering by retry.pdf')).toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('does nothing when the file picker is cancelled', async () => {
  await ready();
  fireEvent.change(uploadInput(), { target: { files: [] } });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it('encodes deletion names, clears deleted selection, and keeps other documents', async () => {
  const name = 'Résumé & #1.pdf';
  const user = await ready([name, 'keep.pdf']);
  await user.click(screen.getByText(name));
  fetchMock.mockResolvedValueOnce(json({ message: 'deleted' }))
    .mockResolvedValueOnce(json({ documents: ['keep.pdf'] }));
  await user.click(within(screen.getByText(name).closest('li')!).getByRole('button'));
  await waitFor(() => expect(screen.queryByText(name)).not.toBeInTheDocument());
  expect(fetchMock.mock.calls[1][0]).toBe(`http://127.0.0.1:8000/documents/${encodeURIComponent(name)}`);
  expect(fetchMock.mock.calls[1][1]!.method).toBe('DELETE');
  expect(screen.getByText('Global Search (All Documents)')).toBeInTheDocument();
  expect(screen.getByText('keep.pdf')).toBeInTheDocument();
});

it('keeps a document and selection when deletion fails', async () => {
  const user = await ready(['keep.pdf']);
  await user.click(screen.getByText('keep.pdf'));
  fetchMock.mockResolvedValueOnce(json({ detail: 'Storage unavailable' }, 503));
  await user.click(within(screen.getByText('keep.pdf').closest('li')!).getByRole('button'));
  expect(await screen.findByRole('alert')).toHaveTextContent('Storage unavailable');
  expect(screen.getByText('Filtering by keep.pdf')).toBeInTheDocument();
  expect(screen.getByText('keep.pdf')).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it('keeps the selected filter when deleting a different document', async () => {
  const user = await ready(['selected.pdf', 'remove.pdf']);
  await user.click(screen.getByText('selected.pdf'));
  fetchMock.mockResolvedValueOnce(json({ message: 'deleted' }))
    .mockResolvedValueOnce(json({ documents: ['selected.pdf'] }));
  await user.click(within(screen.getByText('remove.pdf').closest('li')!).getByRole('button'));
  await waitFor(() => expect(screen.queryByText('remove.pdf')).not.toBeInTheDocument());
  expect(screen.getByText('Filtering by selected.pdf')).toBeInTheDocument();
});

it('reports a list-refresh failure after upload', async () => {
  const user = await ready();
  fetchMock.mockResolvedValueOnce(json({ filename: 'a.pdf', message: 'indexed' }))
    .mockRejectedValueOnce(new Error('Refresh failed'));
  await user.upload(uploadInput(), new File(['pdf'], 'a.pdf', { type: 'application/pdf' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Refresh failed');
  expect(uploadInput()).toBeEnabled();
});

it('lists citations with file and page, and clears them on the next question', async () => {
  const user = await ready(['manual.pdf']);
  await user.type(screen.getByRole('textbox'), 'Pump pressure?');
  fetchMock.mockResolvedValueOnce(json({
    answer: 'It is 4 bar.',
    sources: ['Pump runs at 4 bar.', 'Loose text'],
    citations: [
      { source_file: 'manual.pdf', page: 3, text: 'Pump runs at 4 bar.' },
      { source_file: 'notes.pdf', page: null, text: 'Loose text' },
    ],
  }));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  expect(await screen.findByText('manual.pdf · page 3')).toBeInTheDocument();
  expect(screen.getByText('notes.pdf')).toBeInTheDocument();
  expect(screen.getByText('Pump runs at 4 bar.')).toBeInTheDocument();
  fetchMock.mockResolvedValueOnce(json({ answer: 'No idea.', sources: [] }));
  await user.click(screen.getByRole('button', { name: 'Ask' }));
  await screen.findByText('No idea.');
  expect(screen.queryByText('Sources')).not.toBeInTheDocument();
});
