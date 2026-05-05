export default async function ProfessorTeamPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="mx-auto max-w-3xl py-12 text-center">
      <h1 className="text-2xl font-semibold text-slate-900">Team {id}</h1>
      <p className="mt-2 text-sm text-muted">Coming in Session B.</p>
    </div>
  );
}
