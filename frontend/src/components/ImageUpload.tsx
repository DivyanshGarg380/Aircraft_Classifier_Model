import { useCallback, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
}

export default function ImageUpload({ onFileSelected }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((file: File) => {
    setPreview(URL.createObjectURL(file));
    onFileSelected(file);
  }, [onFileSelected]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors
        ${dragOver ? "border-radar bg-navy-700" : "border-slate-600 bg-navy-800"}`}
    >
      {preview ? (
        <img src={preview} alt="preview" className="mx-auto max-h-64 rounded-lg mb-4" />
      ) : (
        <p className="text-slate-400 mb-4">Drag & drop an aircraft image, or click below</p>
      )}
      <input
        type="file"
        accept="image/*"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        className="text-sm text-slate-300"
      />
    </div>
  );
}